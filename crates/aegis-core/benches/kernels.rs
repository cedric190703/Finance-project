use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(c: &mut Criterion) {
    c.bench_function("version", |b| b.iter(|| aegis_core::VERSION.len()));
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
