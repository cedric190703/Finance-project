//! Criterion benchmarks for the numerical kernels.
//!
//! Run with `cargo bench`. The Monte Carlo group is the one CI watches for
//! regressions; the distribution-function group is there because everything else
//! calls into it, so a slowdown there is a slowdown everywhere.

use aegis_core::black::{self, OptionRight};
use aegis_core::mc::{self, EuropeanOption, Sampler, VarianceReduction};
use aegis_core::normal;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn option() -> EuropeanOption {
    EuropeanOption {
        forward: 100.0,
        strike: 100.0,
        vol: 0.25,
        time: 1.0,
        discount: 0.96,
        right: OptionRight::Call,
    }
}

fn distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal");
    group.bench_function("cdf", |b| b.iter(|| normal::cdf(black_box(0.37))));
    group.bench_function("inverse_cdf", |b| {
        b.iter(|| normal::inverse_cdf(black_box(0.63)))
    });
    group.bench_function("black_price", |b| {
        b.iter(|| black::price(black_box(100.0), 95.0, 0.2, 1.0, 0.97, OptionRight::Call))
    });
    group.finish();
}

fn monte_carlo(c: &mut Criterion) {
    let mut group = c.benchmark_group("mc_european");
    for paths in [1 << 16, 1 << 20] {
        group.throughput(Throughput::Elements(paths as u64));
        for (label, sampler) in [
            ("pseudo", Sampler::PseudoRandom),
            ("quasi", Sampler::QuasiRandom),
        ] {
            group.bench_with_input(BenchmarkId::new(label, paths), &paths, |b, &paths| {
                b.iter(|| {
                    mc::european(
                        black_box(option()),
                        paths,
                        1,
                        sampler,
                        VarianceReduction::recommended(sampler),
                    )
                });
            });
        }
    }
    group.finish();
}

fn batch(c: &mut Criterion) {
    let options: Vec<_> = (60..140)
        .map(|strike| EuropeanOption {
            strike: f64::from(strike),
            ..option()
        })
        .collect();
    let mut group = c.benchmark_group("mc_batch");
    group.throughput(Throughput::Elements(options.len() as u64));
    group.bench_function("80_options_65k_paths", |b| {
        b.iter(|| {
            mc::european_batch(
                black_box(&options),
                1 << 16,
                1,
                Sampler::QuasiRandom,
                VarianceReduction::recommended(Sampler::QuasiRandom),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, distributions, monte_carlo, batch);
criterion_main!(benches);
