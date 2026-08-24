//! The normal distribution, to double precision.
//!
//! The cumulative distribution uses Hart's 1968 rational approximation, which is
//! what most production pricing libraries use: it is accurate to roughly 1e-15
//! across the whole real line, and it is branch-light enough to stay fast in an
//! inner loop.
//!
//! The inverse uses Acklam's algorithm followed by one Halley refinement step.
//! Acklam alone is good to about 1.15e-9 relative; the refinement takes it to
//! full double precision, which matters because the inverse is what turns a
//! quasi-random number in the unit interval into a normal draw. A low-discrepancy
//! sequence pushed through a sloppy inverse is just an expensive random number.

const SQRT_2PI: f64 = 2.506_628_274_631_000_7;

// Acklam's coefficients for the inverse normal, and the breakpoints between
// its three branches. At module scope so the tables are built once.
const A: [f64; 6] = [
    -3.969_683_028_665_376e1,
    2.209_460_984_245_205e2,
    -2.759_285_104_469_687e2,
    1.383_577_518_672_69e2,
    -3.066_479_806_614_716e1,
    2.506_628_277_459_239,
];
const B: [f64; 5] = [
    -5.447_609_879_822_406e1,
    1.615_858_368_580_409e2,
    -1.556_989_798_598_866e2,
    6.680_131_188_771_972e1,
    -1.328_068_155_288_572e1,
];
const C: [f64; 6] = [
    -7.784_894_002_430_293e-3,
    -3.223_964_580_411_365e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
];
const D: [f64; 4] = [
    7.784_695_709_041_462e-3,
    3.224_671_290_700_398e-1,
    2.445_134_137_142_996,
    3.754_408_661_907_416,
];

const LOW: f64 = 0.024_25;
const HIGH: f64 = 1.0 - LOW;

/// Standard normal probability density.
#[inline]
#[must_use]
pub fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

/// Standard normal cumulative distribution, via Hart's rational approximation.
#[must_use]
pub fn cdf(x: f64) -> f64 {
    let z = x.abs();
    let upper = if z > 37.0 {
        0.0
    } else {
        let e = (-0.5 * z * z).exp();
        if z < 7.071_067_811_865_475 {
            let numerator = (((((3.526_249_659_989_109e-2 * z + 0.700_383_064_443_688) * z
                + 6.373_962_203_531_65)
                * z
                + 33.912_866_078_383)
                * z
                + 112.079_291_497_871)
                * z
                + 221.213_596_169_931)
                * z
                + 220.206_867_912_376;
            let denominator = ((((((8.838_834_764_831_844e-2 * z + 1.755_667_163_182_642) * z
                + 16.064_177_579_206_95)
                * z
                + 86.780_732_202_946_32)
                * z
                + 296.564_248_779_674)
                * z
                + 637.333_633_378_831)
                * z
                + 793.826_512_519_948)
                * z
                + 440.413_735_824_752;
            e * numerator / denominator
        } else {
            let continued = z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 0.65))));
            e / (continued * SQRT_2PI)
        }
    };

    if x > 0.0 {
        1.0 - upper
    } else {
        upper
    }
}

/// Inverse standard normal cumulative distribution.
///
/// # Panics
/// Never panics; values outside `(0, 1)` map to infinities, matching the limits.
#[must_use]
pub fn inverse_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let mut x = if p < LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley step against the (much more accurate) CDF above.
    let error = cdf(x) - p;
    let density = pdf(x);
    if density > 0.0 {
        let u = error / density;
        x -= u / (1.0 + 0.5 * x * u);
    }
    x
}

/// Two-sided 97.5% Student-t quantile for a given number of degrees of freedom.
///
/// Tabulated for small samples and falling back to the normal quantile past
/// thirty, where the difference is below half a percent. It exists so that a
/// standard error estimated from a handful of randomised replicas is turned into
/// an honest confidence interval rather than an optimistic one.
#[must_use]
pub fn student_t_975(degrees_of_freedom: usize) -> f64 {
    const TABLE: [f64; 31] = [
        f64::INFINITY, // 0 df: no interval can be formed
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    ];
    TABLE
        .get(degrees_of_freedom)
        .copied()
        .unwrap_or(1.959_963_984_540_054)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn cdf_matches_known_values() {
        assert_abs_diff_eq!(cdf(0.0), 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(cdf(1.0), 0.841_344_746_068_543, epsilon = 1e-14);
        assert_abs_diff_eq!(cdf(-1.96), 0.024_997_895_148_220_43, epsilon = 1e-14);
        assert_abs_diff_eq!(cdf(5.0), 0.999_999_713_348_428_2, epsilon = 1e-14);
    }

    #[test]
    fn cdf_is_symmetric() {
        for i in 0..400 {
            let x = f64::from(i) * 0.02;
            assert_abs_diff_eq!(cdf(x) + cdf(-x), 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn inverse_round_trips_the_cdf() {
        for i in 1..1000 {
            let p = f64::from(i) / 1000.0;
            assert_abs_diff_eq!(cdf(inverse_cdf(p)), p, epsilon = 1e-14);
        }
    }

    #[test]
    fn student_t_quantiles_shrink_towards_the_normal_one() {
        assert!(student_t_975(0).is_infinite());
        assert_abs_diff_eq!(student_t_975(1), 12.706, epsilon = 1e-3);
        assert_abs_diff_eq!(student_t_975(7), 2.365, epsilon = 1e-3);
        assert_abs_diff_eq!(student_t_975(1000), 1.959_963_984_540_054, epsilon = 1e-12);
        for df in 1..30 {
            assert!(student_t_975(df) > student_t_975(df + 1));
        }
    }

    #[test]
    fn inverse_handles_the_tails() {
        assert!(inverse_cdf(0.0).is_infinite());
        assert!(inverse_cdf(1.0).is_infinite());
        assert_abs_diff_eq!(inverse_cdf(0.5), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(inverse_cdf(0.975), 1.959_963_984_540_054, epsilon = 1e-10);
    }
}
