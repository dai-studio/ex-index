//! Micro-benchmarks of the distance kernels (`cargo bench --bench distance`).
use criterion::{criterion_group, criterion_main, Criterion};
use ex_index_core::distance::{dot, l2sq};
use ex_index_core::harness::gaussian_vectors;
use std::hint::black_box;

fn kernels(c: &mut Criterion) {
    for d in [128usize, 768] {
        let v = gaussian_vectors(2, d, 1);
        let (a, b) = v.split_at(d);
        c.bench_function(&format!("l2sq/{d}"), |bch| {
            bch.iter(|| l2sq(black_box(a), black_box(b)))
        });
        c.bench_function(&format!("dot/{d}"), |bch| {
            bch.iter(|| dot(black_box(a), black_box(b)))
        });
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);
