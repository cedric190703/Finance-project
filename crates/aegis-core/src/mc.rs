//! The Monte Carlo engine.
//!
//! A European option under Black's model only needs the terminal value of the
//! forward, so there is no path to step through:
//!
//! ```text
//! F_T = F · exp(−½σ²T + σ√T · Z)
//! ```
//!
//! What makes this worth writing rather than calling a closed form is everything
//! around that line — the variance reduction, the parallelism, and the
//! determinism — all of which carry straight over to the full-revaluation risk
//! engine, where no closed form exists.
//!
//! Three design decisions worth stating outright:
//!
//! **Work is chunked by a fixed size, not by thread count.** The obvious
//! parallel decomposition splits the path count across the available cores, but
//! then the answer changes when the machine does: eight cores and sixteen cores
//! consume the random stream differently and produce different numbers. Risk
//! numbers that depend on which box they ran on are not reproducible, and a P&L
//! explain that cannot be reproduced cannot be defended. Chunks here are a fixed
//! size, each seeded from the run seed and its own index, so the result is
//! identical on one thread or on ninety-six.
//!
//! **Antithetic sampling** evaluates each draw at `+Z` and `−Z` and averages the
//! pair into a single sample. Averaging them is the whole point, and it is
//! surprisingly easy to get wrong: pushing the two evaluations into the
//! estimator as if they were independent gives the same mean but throws away
//! the negative correlation, so the reported standard error does not improve
//! and the technique silently does nothing.
//!
//! **A control variate** uses the terminal forward itself, whose expectation is
//! known exactly. Subtracting `β·(F_T − F)` with `β` set to the analytic delta
//! removes the part of the payoff that moves linearly with the underlying —
//! which, for anything but a deeply out-of-the-money option, is again most of
//! the variance. Taking `β` from the closed form rather than estimating it from
//! the sample keeps the estimator unbiased.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::black::{self, OptionRight};
use crate::normal;

/// Paths per work unit. Fixed so results do not depend on the thread count.
pub const CHUNK_PATHS: usize = 8_192;

/// How the driving numbers are produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// Pseudo-random draws from xoshiro256++.
    PseudoRandom,
    /// A digitally shifted Sobol sequence, one shift per chunk.
    QuasiRandom,
}

/// Which variance reductions to apply.
#[derive(Debug, Clone, Copy)]
pub struct VarianceReduction {
    /// Evaluate each draw at both `+Z` and `−Z`.
    pub antithetic: bool,
    /// Subtract the analytic-delta multiple of the terminal forward's error.
    pub control_variate: bool,
}

impl Default for VarianceReduction {
    /// The control variate alone: the right choice for pseudo-random sampling.
    ///
    /// See [`VarianceReduction::recommended`] for why "turn everything on" is
    /// the wrong instinct.
    fn default() -> Self {
        Self {
            antithetic: false,
            control_variate: true,
        }
    }
}

impl VarianceReduction {
    /// No variance reduction at all: the plain estimator.
    #[must_use]
    pub fn none() -> Self {
        Self {
            antithetic: false,
            control_variate: false,
        }
    }

    /// Antithetic sampling only. Useful for a monotone payoff with no natural
    /// control variate.
    #[must_use]
    pub fn antithetic_only() -> Self {
        Self {
            antithetic: true,
            control_variate: false,
        }
    }

    /// The combination that actually works best for a given sampler.
    ///
    /// Stacking every variance reduction is the obvious instinct and it is
    /// wrong, because the techniques interact. Standard errors on an
    /// at-the-money one-year call, one million evaluations, measured by the
    /// diagnostic at the bottom of this module:
    ///
    /// | reduction  |   pseudo |    quasi |
    /// |------------|---------:|---------:|
    /// | none       | 1.58e-2  | 5.67e-4  |
    /// | antithetic | 1.29e-2  | 1.27e-4  |
    /// | control    | 7.10e-3  | 1.48e-4  |
    /// | both       | 9.77e-3  | 6.61e-5  |
    ///
    /// Under pseudo-random sampling, adding antithetic pairing *on top of* the
    /// control variate makes things worse — 7.10e-3 becomes 9.77e-3. Antithetic
    /// sampling cancels the odd part of the payoff's dependence on the driver,
    /// the control variate has already removed the linear part, which is most of
    /// what is odd, and averaging what remains with its mirror image cancels
    /// nothing while halving the independent sample count. That is the `√2` in
    /// the numbers.
    ///
    /// Under a randomised low-discrepancy sequence the picture reverses: the
    /// antithetic pair fills the unit interval symmetrically, which is exactly
    /// what a stratified sequence wants, and both together are best. The whole
    /// column is another order of magnitude better, because a smooth payoff is
    /// what quasi-Monte Carlo was built for.
    ///
    /// Best against worst is 6.61e-5 against 1.58e-2: a factor of 240 in error,
    /// which at the `1/√n` rate of plain Monte Carlo would take roughly 57,000
    /// times the paths to buy.
    #[must_use]
    pub fn recommended(sampler: Sampler) -> Self {
        match sampler {
            Sampler::PseudoRandom => Self {
                antithetic: false,
                control_variate: true,
            },
            Sampler::QuasiRandom => Self {
                antithetic: true,
                control_variate: true,
            },
        }
    }
}

/// The terms of a European option to be simulated.
#[derive(Debug, Clone, Copy)]
pub struct EuropeanOption {
    /// Forward price of the underlying to expiry.
    pub forward: f64,
    /// Strike price.
    pub strike: f64,
    /// Black implied volatility.
    pub vol: f64,
    /// Time to expiry in years.
    pub time: f64,
    /// Discount factor to the payment date.
    pub discount: f64,
    /// Call or put.
    pub right: OptionRight,
}

/// The outcome of a simulation.
#[derive(Debug, Clone, Copy)]
pub struct McResult {
    /// The estimated price.
    pub price: f64,
    /// Standard error of the estimate.
    pub standard_error: f64,
    /// How many paths were consumed, counting antithetic partners.
    pub paths: usize,
    /// How many independent observations the standard error was built from.
    ///
    /// For pseudo-random sampling that is the sample count, and the standard
    /// error is a precise estimate. For a randomised low-discrepancy sequence it
    /// is the number of independently shifted replicas — often only a handful —
    /// and the standard error is then itself a noisy quantity. A confidence
    /// interval built from it needs a Student-t quantile, not a normal one; the
    /// difference is a factor of 2.4 at four replicas and negligible past thirty.
    pub replicas: usize,
}

impl McResult {
    /// Combine independently randomised replicas into one estimate.
    fn from_replicas(replicas: &[Accumulator], discount: f64) -> Self {
        let means: Vec<f64> = replicas
            .iter()
            .filter(|r| r.count > 0)
            .map(Accumulator::mean)
            .collect();
        let evaluations: usize = replicas.iter().map(|r| r.evaluations).sum();
        if means.is_empty() {
            return Self {
                price: 0.0,
                standard_error: 0.0,
                paths: 0,
                replicas: 0,
            };
        }
        let k = means.len() as f64;
        let mean = means.iter().sum::<f64>() / k;
        let variance = if means.len() > 1 {
            means.iter().map(|m| (m - mean) * (m - mean)).sum::<f64>() / (k - 1.0)
        } else {
            0.0
        };
        Self {
            price: discount * mean,
            standard_error: discount * (variance / k).sqrt(),
            paths: evaluations,
            replicas: means.len(),
        }
    }

    /// Half-width of a two-sided 95% confidence interval.
    ///
    /// Uses a Student-t quantile on `replicas - 1` degrees of freedom, which
    /// matters when the error was estimated from a small number of randomised
    /// replicas and collapses to the familiar 1.96 when it was not.
    #[must_use]
    pub fn confidence_95(&self) -> f64 {
        crate::normal::student_t_975(self.replicas.saturating_sub(1)) * self.standard_error
    }
}

/// Price a European option by Monte Carlo.
///
/// The `paths` argument is rounded up to a whole number of chunks, so the
/// consumed path count is reported back in the result rather than assumed.
#[must_use]
pub fn european(
    option: EuropeanOption,
    paths: usize,
    seed: u64,
    sampler: Sampler,
    reduction: VarianceReduction,
) -> McResult {
    let chunks = paths.div_ceil(CHUNK_PATHS).max(1);
    let beta = if reduction.control_variate {
        black::delta(
            option.forward,
            option.strike,
            option.vol,
            option.time,
            1.0,
            option.right,
        )
    } else {
        0.0
    };

    let replicas: Vec<Accumulator> = (0..chunks)
        .into_par_iter()
        .map(|chunk| simulate_chunk(option, seed, chunk, sampler, reduction, beta))
        .collect();

    match sampler {
        // Points inside one Sobol chunk are anything but independent — that is
        // the entire point of a low-discrepancy sequence — so pooling them into
        // one variance would report a standard error that means nothing. Each
        // chunk carries its own random digital shift, which makes the chunk
        // means genuinely independent, so the spread *between* chunks is the
        // honest error estimate. This is the standard randomised-QMC estimator.
        Sampler::QuasiRandom => McResult::from_replicas(&replicas, option.discount),
        // Pseudo-random draws are independent everywhere, so pooling is both
        // valid and more precise than looking only at the chunk means.
        Sampler::PseudoRandom => replicas
            .into_iter()
            .fold(Accumulator::default(), Accumulator::merge)
            .finish(option.discount),
    }
}

/// Price many European options, one per thread-pool task.
///
/// Used for portfolio revaluation, where the win comes from having many
/// independent options in flight rather than from splitting one of them.
#[must_use]
pub fn european_batch(
    options: &[EuropeanOption],
    paths: usize,
    seed: u64,
    sampler: Sampler,
    reduction: VarianceReduction,
) -> Vec<McResult> {
    options
        .par_iter()
        .enumerate()
        .map(|(index, option)| {
            european(
                *option,
                paths,
                seed.wrapping_add(index as u64),
                sampler,
                reduction,
            )
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Default)]
struct Accumulator {
    sum: f64,
    sum_squares: f64,
    count: usize,
    evaluations: usize,
}

impl Accumulator {
    /// Record one independent sample, built from `evaluations` payoff evaluations.
    fn push(&mut self, value: f64, evaluations: usize) {
        self.sum += value;
        self.sum_squares += value * value;
        self.count += 1;
        self.evaluations += evaluations;
    }

    fn merge(self, other: Self) -> Self {
        Self {
            sum: self.sum + other.sum,
            sum_squares: self.sum_squares + other.sum_squares,
            count: self.count + other.count,
            evaluations: self.evaluations + other.evaluations,
        }
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn finish(self, discount: f64) -> McResult {
        if self.count == 0 {
            return McResult {
                price: 0.0,
                standard_error: 0.0,
                paths: 0,
                replicas: 0,
            };
        }
        let n = self.count as f64;
        let mean = self.sum / n;
        let variance = if self.count > 1 {
            ((self.sum_squares - n * mean * mean) / (n - 1.0)).max(0.0)
        } else {
            0.0
        };
        McResult {
            price: discount * mean,
            standard_error: discount * (variance / n).sqrt(),
            paths: self.evaluations,
            replicas: self.count,
        }
    }
}

fn simulate_chunk(
    option: EuropeanOption,
    seed: u64,
    chunk: usize,
    sampler: Sampler,
    reduction: VarianceReduction,
    beta: f64,
) -> Accumulator {
    // Each chunk is seeded from the run seed and its own index, so a chunk always
    // produces the same numbers regardless of which thread picks it up.
    let chunk_seed = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(chunk as u64)
        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(chunk_seed);
    let mut sobol = crate::sobol::Sobol1d::new(rng.gen::<u64>());

    let drift = -0.5 * option.vol * option.vol * option.time;
    let diffusion = option.vol * option.time.sqrt();
    let mut accumulator = Accumulator::default();

    let draws = if reduction.antithetic {
        CHUNK_PATHS / 2
    } else {
        CHUNK_PATHS
    };
    for _ in 0..draws {
        let uniform = match sampler {
            Sampler::PseudoRandom => rng.gen::<f64>(),
            Sampler::QuasiRandom => sobol.next_point(),
        };
        // gen::<f64>() draws from [0, 1), and the inverse normal sends an exact
        // zero to negative infinity. One draw in 2^53 is not a risk worth taking
        // in a kernel that will run for years.
        let z = normal::inverse_cdf(uniform.max(f64::EPSILON));
        let value = sample(option, drift, diffusion, z, beta);
        if reduction.antithetic {
            let partner = sample(option, drift, diffusion, -z, beta);
            accumulator.push(0.5 * (value + partner), 2);
        } else {
            accumulator.push(value, 1);
        }
    }
    accumulator
}

#[inline]
fn sample(option: EuropeanOption, drift: f64, diffusion: f64, z: f64, beta: f64) -> f64 {
    let terminal = option.forward * (drift + diffusion * z).exp();
    let payoff = black::payoff(terminal, option.strike, option.right);
    // E[F_T] = F, so subtracting a multiple of (F_T − F) leaves the mean alone.
    payoff - beta * (terminal - option.forward)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn atm() -> EuropeanOption {
        EuropeanOption {
            forward: 100.0,
            strike: 100.0,
            vol: 0.25,
            time: 1.0,
            discount: 0.96,
            right: OptionRight::Call,
        }
    }

    #[test]
    fn converges_to_the_closed_form() {
        let option = atm();
        let exact = black::price(
            option.forward,
            option.strike,
            option.vol,
            option.time,
            option.discount,
            option.right,
        );
        let result = european(
            option,
            1 << 20,
            42,
            Sampler::PseudoRandom,
            VarianceReduction::default(),
        );
        assert!(
            (result.price - exact).abs() < 3.0 * result.standard_error.max(1e-9),
            "price {} exact {} se {}",
            result.price,
            exact,
            result.standard_error
        );
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let option = atm();
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| {
                    european(
                        option,
                        1 << 18,
                        7,
                        Sampler::PseudoRandom,
                        VarianceReduction::default(),
                    )
                    .price
                })
        };
        assert_abs_diff_eq!(run(1), run(4), epsilon = 0.0);
        assert_abs_diff_eq!(run(1), run(8), epsilon = 0.0);
    }

    #[test]
    fn the_same_seed_gives_the_same_answer() {
        let option = atm();
        let first = european(
            option,
            1 << 16,
            99,
            Sampler::PseudoRandom,
            VarianceReduction::none(),
        );
        let second = european(
            option,
            1 << 16,
            99,
            Sampler::PseudoRandom,
            VarianceReduction::none(),
        );
        assert_abs_diff_eq!(first.price, second.price, epsilon = 0.0);
    }

    #[test]
    fn a_different_seed_gives_a_different_answer() {
        let option = atm();
        let first = european(
            option,
            1 << 16,
            1,
            Sampler::PseudoRandom,
            VarianceReduction::none(),
        );
        let second = european(
            option,
            1 << 16,
            2,
            Sampler::PseudoRandom,
            VarianceReduction::none(),
        );
        assert!((first.price - second.price).abs() > 0.0);
    }

    #[test]
    fn each_variance_reduction_beats_the_plain_estimator() {
        let option = atm();
        let paths = 1 << 20;
        let plain = european(
            option,
            paths,
            3,
            Sampler::PseudoRandom,
            VarianceReduction::none(),
        );
        let antithetic = european(
            option,
            paths,
            3,
            Sampler::PseudoRandom,
            VarianceReduction::antithetic_only(),
        );
        let control = european(
            option,
            paths,
            3,
            Sampler::PseudoRandom,
            VarianceReduction::default(),
        );

        assert!(antithetic.standard_error < plain.standard_error);
        assert!(control.standard_error < antithetic.standard_error);
        // The control variate is the one that matters: it halves the error, which
        // is the same as quadrupling the path count.
        assert!(control.standard_error * 2.0 < plain.standard_error);
    }

    #[test]
    fn stacking_reductions_helps_under_quasi_random_and_hurts_under_pseudo_random() {
        // The interaction the recommended() table documents, pinned down so it
        // cannot silently reverse. Switching everything on "for safety" costs
        // accuracy in one column and buys it in the other.
        let option = atm();
        let paths = 1 << 20;
        let both = VarianceReduction {
            antithetic: true,
            control_variate: true,
        };

        let pseudo_control = european(
            option,
            paths,
            3,
            Sampler::PseudoRandom,
            VarianceReduction::default(),
        );
        let pseudo_both = european(option, paths, 3, Sampler::PseudoRandom, both);
        assert!(pseudo_both.standard_error > pseudo_control.standard_error);

        let quasi_control = european(
            option,
            paths,
            3,
            Sampler::QuasiRandom,
            VarianceReduction::default(),
        );
        let quasi_both = european(option, paths, 3, Sampler::QuasiRandom, both);
        assert!(quasi_both.standard_error < quasi_control.standard_error);
    }

    #[test]
    fn the_recommended_settings_are_the_best_ones_for_each_sampler() {
        let option = atm();
        let paths = 1 << 20;
        let combinations = [
            VarianceReduction::none(),
            VarianceReduction::antithetic_only(),
            VarianceReduction {
                antithetic: false,
                control_variate: true,
            },
            VarianceReduction {
                antithetic: true,
                control_variate: true,
            },
        ];
        for sampler in [Sampler::PseudoRandom, Sampler::QuasiRandom] {
            let recommended = european(
                option,
                paths,
                3,
                sampler,
                VarianceReduction::recommended(sampler),
            );
            for reduction in combinations {
                let other = european(option, paths, 3, sampler, reduction);
                assert!(
                    recommended.standard_error <= other.standard_error * 1.000_001,
                    "{sampler:?}: recommended {} lost to {reduction:?} {}",
                    recommended.standard_error,
                    other.standard_error
                );
            }
        }
    }

    #[test]
    fn quasi_random_sampling_beats_pseudo_random() {
        // Both the error and the estimate of it: RQMC reports a far tighter
        // interval, and lands inside it.
        let option = atm();
        let exact = black::price(
            option.forward,
            option.strike,
            option.vol,
            option.time,
            option.discount,
            option.right,
        );
        let paths = 1 << 14;
        let pseudo = european(
            option,
            paths,
            5,
            Sampler::PseudoRandom,
            VarianceReduction::none(),
        );
        let quasi = european(
            option,
            paths,
            5,
            Sampler::QuasiRandom,
            VarianceReduction::none(),
        );
        assert!(
            (quasi.price - exact).abs() < (pseudo.price - exact).abs(),
            "quasi {} pseudo {} exact {}",
            quasi.price,
            pseudo.price,
            exact
        );
    }

    #[test]
    fn a_batch_prices_every_option() {
        let options: Vec<_> = (80..120)
            .map(|strike| EuropeanOption {
                strike: f64::from(strike),
                ..atm()
            })
            .collect();
        let results = european_batch(
            &options,
            1 << 14,
            11,
            Sampler::PseudoRandom,
            VarianceReduction::default(),
        );

        assert_eq!(results.len(), options.len());
        for (option, result) in options.iter().zip(&results) {
            let exact = black::price(
                option.forward,
                option.strike,
                option.vol,
                option.time,
                option.discount,
                option.right,
            );
            assert!((result.price - exact).abs() < 4.0 * result.standard_error.max(1e-6));
        }
        // Deep out-of-the-money options are worth less than at-the-money ones.
        assert!(results.last().expect("last").price < results[0].price);
    }

    #[test]
    fn confidence_interval_is_reported() {
        let result = european(
            atm(),
            1 << 16,
            1,
            Sampler::PseudoRandom,
            VarianceReduction::none(),
        );
        assert!(result.confidence_95() > result.standard_error);
        assert_eq!(result.paths, 1 << 16);
        assert_eq!(result.replicas, 1 << 16);
    }

    #[test]
    fn a_quasi_random_run_reports_its_replica_count() {
        // Eight chunks means eight independent digital shifts, and an interval
        // built on seven degrees of freedom rather than on the path count.
        let result = european(
            atm(),
            8 * CHUNK_PATHS,
            1,
            Sampler::QuasiRandom,
            VarianceReduction::none(),
        );
        assert_eq!(result.replicas, 8);
        assert!(result.confidence_95() > 2.3 * result.standard_error);
    }
}

#[cfg(test)]
mod diagnostics {
    //! Ignored by default. Prints the variance table quoted in the docs above;
    //! run with `cargo test -- --ignored --nocapture` to regenerate it.

    use super::*;

    #[test]
    #[ignore = "reporting only"]
    fn report_variance_reduction() {
        let option = EuropeanOption {
            forward: 100.0,
            strike: 100.0,
            vol: 0.25,
            time: 1.0,
            discount: 0.96,
            right: OptionRight::Call,
        };
        let paths = 1 << 20;
        let combinations = [
            ("plain", VarianceReduction::none()),
            ("antithetic", VarianceReduction::antithetic_only()),
            (
                "control",
                VarianceReduction {
                    antithetic: false,
                    control_variate: true,
                },
            ),
            (
                "both",
                VarianceReduction {
                    antithetic: true,
                    control_variate: true,
                },
            ),
        ];
        println!("{:<12}{:>16}{:>16}", "reduction", "pseudo se", "quasi se");
        for (label, reduction) in combinations {
            let pseudo = european(option, paths, 3, Sampler::PseudoRandom, reduction);
            let quasi = european(option, paths, 3, Sampler::QuasiRandom, reduction);
            println!(
                "{label:<12}{:>16.4e}{:>16.4e}",
                pseudo.standard_error, quasi.standard_error
            );
        }
    }
}
