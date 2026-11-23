//! Write-ahead log of un-merged upserts / deletes.
//!
//! One file per generation (`wal.<gen>.log`). Record framing:
//! `[u32 payload_len][u32 crc32(payload)][payload]`, payload:
//! * upsert: `u8 1, u16 id_len, id, u32 dim, f32 × dim, u32 meta_len, meta JSON`
//! * delete: `u8 2, u16 id_len, id`
//!
//! A batch is appended with one `write_all` and fsynced before the caller
//! acknowledges. Replay stops at the first torn / corrupt record and the file
//! is truncated there.

use crate::error::{Error, Result};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum WalOp {
    Upsert {
        id: String,
        vector: Vec<f32>,
        metadata: Option<Vec<u8>>,
    },
    Delete {
        id: String,
    },
}

impl WalOp {
    pub fn id(&self) -> &str {
        match self {
            WalOp::Upsert { id, .. } | WalOp::Delete { id } => id,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&[0u8; 8]);
        match self {
            WalOp::Upsert {
                id,
                vector,
                metadata,
            } => {
                out.push(1);
                out.extend_from_slice(&(id.len() as u16).to_le_bytes());
                out.extend_from_slice(id.as_bytes());
                out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                for x in vector {
                    out.extend_from_slice(&x.to_le_bytes());
                }
                let m = metadata.as_deref().unwrap_or(&[]);
                out.extend_from_slice(&(m.len() as u32).to_le_bytes());
                out.extend_from_slice(m);
            }
            WalOp::Delete { id } => {
                out.push(2);
                out.extend_from_slice(&(id.len() as u16).to_le_bytes());
                out.extend_from_slice(id.as_bytes());
            }
        }
        let len = (out.len() - start - 8) as u32;
        let crc = crate::util::crc32(&out[start + 8..]);
        out[start..start + 4].copy_from_slice(&len.to_le_bytes());
        out[start + 4..start + 8].copy_from_slice(&crc.to_le_bytes());
    }

    fn decode(p: &[u8]) -> Option<WalOp> {
        let mut c = Cursor { b: p, pos: 0 };
        let op = c.u8()?;
        let idlen = c.u16()? as usize;
        let id = String::from_utf8(c.take(idlen)?.to_vec()).ok()?;
        let out = match op {
            1 => {
                let dim = c.u32()? as usize;
                let raw = c.take(dim.checked_mul(4)?)?;
                let vector = raw
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let ml = c.u32()? as usize;
                let m = c.take(ml)?;
                let metadata = if ml == 0 { None } else { Some(m.to_vec()) };
                WalOp::Upsert {
                    id,
                    vector,
                    metadata,
                }
            }
            2 => WalOp::Delete { id },
            _ => return None,
        };
        (c.pos == p.len()).then_some(out)
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

pub fn wal_path(dir: &Path, gen: u64) -> PathBuf {
    dir.join(format!("wal.{gen}.log"))
}

/// Parse `wal.<gen>.log` file names.
pub fn parse_wal_name(name: &str) -> Option<u64> {
    name.strip_prefix("wal.")?
        .strip_suffix(".log")?
        .parse()
        .ok()
}

/// Appender for the active WAL generation.
pub struct WalWriter {
    file: File,
    path: PathBuf,
    gen: u64,
}

impl WalWriter {
    /// Open (creating if needed) for append.
    pub fn open(dir: &Path, gen: u64) -> Result<WalWriter> {
        let path = wal_path(dir, gen);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        crate::util::fsync_dir(dir)?;
        Ok(WalWriter { file, path, gen })
    }

    pub fn gen(&self) -> u64 {
        self.gen
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a batch and fsync before returning.
    pub fn append(&mut self, ops: &[WalOp]) -> Result<()> {
        let mut buf = Vec::new();
        for op in ops {
            op.encode(&mut buf);
        }
        self.file.write_all(&buf)?;
        self.file.sync_data()?;
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

/// Read every valid record; truncates a torn tail. Missing file → empty.
pub fn replay(path: &Path) -> Result<Vec<WalOp>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let total = file.metadata()?.len();
    let mut r = BufReader::new(file);
    let mut ops = Vec::new();
    let mut valid: u64 = 0;
    let mut hdr = [0u8; 8];
    loop {
        if r.read_exact(&mut hdr).is_err() {
            break;
        }
        let len = u32::from_le_bytes(hdr[..4].try_into().unwrap()) as u64;
        let crc = u32::from_le_bytes(hdr[4..].try_into().unwrap());
        if valid + 8 + len > total {
            break;
        }
        let mut payload = vec![0u8; len as usize];
        if r.read_exact(&mut payload).is_err() || crate::util::crc32(&payload) != crc {
            break;
        }
        match WalOp::decode(&payload) {
            Some(op) => ops.push(op),
            None => break,
        }
        valid += 8 + len;
    }
    if valid < total {
        tracing::warn!(path = %path.display(), valid, total, "truncating torn WAL tail");
        let f = OpenOptions::new().write(true).open(path)?;
        f.set_len(valid)?;
        f.sync_all()?;
    }
    Ok(ops)
}

/// Sorted WAL generations present in `dir`.
pub fn list_generations(dir: &Path) -> Result<Vec<u64>> {
    let mut gens = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        if let Some(g) = e.file_name().to_str().and_then(parse_wal_name) {
            gens.push(g);
        }
    }
    gens.sort_unstable();
    Ok(gens)
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Error::Corrupt(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_replay_and_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let ops = vec![
            WalOp::Upsert {
                id: "a".into(),
                vector: vec![1.0, -2.5],
                metadata: Some(br#"{"k":1}"#.to_vec()),
            },
            WalOp::Upsert {
                id: "β".into(),
                vector: vec![0.0, 3.0],
                metadata: None,
            },
            WalOp::Delete { id: "a".into() },
        ];
        {
            let mut w = WalWriter::open(dir.path(), 3).unwrap();
            w.append(&ops[..2]).unwrap();
            w.append(&ops[2..]).unwrap();
        }
        let p = wal_path(dir.path(), 3);
        assert_eq!(replay(&p).unwrap(), ops);
        // simulate a torn write
        let full = std::fs::metadata(&p).unwrap().len();
        let mut f = OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(&[9, 0, 0, 0, 1, 2]).unwrap();
        drop(f);
        assert_eq!(replay(&p).unwrap(), ops);
        assert_eq!(std::fs::metadata(&p).unwrap().len(), full);
        // corrupt a byte in the last record → it is dropped
        let mut bytes = std::fs::read(&p).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF;
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(replay(&p).unwrap(), ops[..2].to_vec());
        assert_eq!(list_generations(dir.path()).unwrap(), vec![3]);
        assert_eq!(replay(&wal_path(dir.path(), 9)).unwrap(), vec![]);
    }
}
