//! Growable bitset used for tombstones.

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Bitset {
    words: Vec<u64>,
    len: usize,
    ones: usize,
}

impl Bitset {
    pub fn new(len: usize) -> Self {
        Bitset {
            words: vec![0; len.div_ceil(64)],
            len,
            ones: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn count_ones(&self) -> usize {
        self.ones
    }

    pub fn grow(&mut self, len: usize) {
        if len > self.len {
            self.len = len;
            self.words.resize(len.div_ceil(64), 0);
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        i < self.len && (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Set bit `i` (growing if needed). Returns true if it was newly set.
    pub fn set(&mut self, i: usize) -> bool {
        self.grow(i + 1);
        let w = &mut self.words[i / 64];
        let mask = 1u64 << (i % 64);
        if *w & mask == 0 {
            *w |= mask;
            self.ones += 1;
            true
        } else {
            false
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.words.len() * 8);
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        for w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<Bitset> {
        if b.len() < 8 {
            return None;
        }
        let len = u64::from_le_bytes(b[..8].try_into().ok()?) as usize;
        let nw = len.div_ceil(64);
        if b.len() != 8 + nw * 8 {
            return None;
        }
        let words: Vec<u64> = b[8..]
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let ones = words.iter().map(|w| w.count_ones() as usize).sum();
        Some(Bitset { words, len, ones })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let mut b = Bitset::new(10);
        assert!(b.set(3));
        assert!(!b.set(3));
        assert!(b.set(130));
        assert_eq!(b.count_ones(), 2);
        assert!(b.get(130) && b.get(3) && !b.get(4) && !b.get(1000));
        let c = Bitset::from_bytes(&b.to_bytes()).unwrap();
        assert_eq!(b, c);
    }
}
