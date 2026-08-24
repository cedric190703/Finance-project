//! A one-dimensional Sobol sequence, for quasi-Monte Carlo.
//!
//! Pseudo-random Monte Carlo converges at `1/√n`: to halve the error you need
//! four times the paths. A low-discrepancy sequence fills the unit interval more
//! evenly than randomness manages and converges closer to `1/n`, which for a
//! smooth payoff is the difference between a million paths and ten thousand.
//!
//! In one dimension the Sobol sequence is the van der Corput sequence in base
//! two: reverse the bits of the index and read them as a fraction. The Gray-code
//! construction here produces the same point set incrementally, one XOR per
//! draw.
//!
//! A raw low-discrepancy sequence is deterministic, which means it has no
//! standard error to report — a point estimate with no error bar is not much use
//! to a risk report. The usual answer, and the one used here, is a random digital
//! shift: XOR every point with one random word. That preserves the equidistribution
//! properties while making the estimator unbiased, so running a handful of
//! independently shifted replicas gives back an honest confidence interval.

/// A scrambled one-dimensional Sobol generator.
#[derive(Debug, Clone)]
pub struct Sobol1d {
    index: u64,
    state: u64,
    shift: u64,
}

const SCALE: f64 = 1.0 / (1u128 << 64) as f64;

impl Sobol1d {
    /// Create a generator with the given random digital shift.
    #[must_use]
    pub fn new(shift: u64) -> Self {
        Self {
            index: 0,
            state: 0,
            shift,
        }
    }

    /// Return the next point in `(0, 1)`.
    #[must_use]
    pub fn next_point(&mut self) -> f64 {
        self.index += 1;
        // The direction number for dimension one is 2^(64-j); the Gray-code
        // update flips exactly the bit belonging to the lowest zero of the index.
        let bit = self.index.trailing_zeros();
        self.state ^= 1u64 << (63 - bit);
        let value = (self.state ^ self.shift) as f64 * SCALE;
        // Keep the result strictly inside the interval: the inverse normal maps
        // an exact zero to negative infinity.
        value.clamp(f64::EPSILON, 1.0 - f64::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshifted_sequence_starts_with_the_van_der_corput_points() {
        let mut sobol = Sobol1d::new(0);
        let expected = [0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125];
        for want in expected {
            let got = sobol.next_point();
            assert!((got - want).abs() < 1e-15, "got {got}, want {want}");
        }
    }

    #[test]
    fn points_stay_inside_the_open_unit_interval() {
        let mut sobol = Sobol1d::new(0x9e37_79b9_7f4a_7c15);
        for _ in 0..10_000 {
            let point = sobol.next_point();
            assert!(point > 0.0 && point < 1.0);
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn the_sequence_is_far_more_uniform_than_randomness() {
        // Fill 1024 points and check every one of 32 equal buckets gets exactly
        // its share. A pseudo-random draw would not manage that.
        let mut sobol = Sobol1d::new(0);
        let mut buckets = [0usize; 32];
        for _ in 0..1024 {
            let point = sobol.next_point();
            buckets[(point * 32.0) as usize] += 1;
        }
        assert!(buckets.iter().all(|&count| count == 32));
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn a_digital_shift_preserves_equidistribution() {
        let mut sobol = Sobol1d::new(0xdead_beef_cafe_1234);
        let mut buckets = [0usize; 32];
        for _ in 0..1024 {
            buckets[(sobol.next_point() * 32.0) as usize] += 1;
        }
        assert!(buckets.iter().all(|&count| count == 32));
    }
}
