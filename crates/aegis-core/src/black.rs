//! Black's formula in forward form, mirroring the Python implementation.
//!
//! It is here for two reasons. It supplies the control variate coefficient the
//! Monte Carlo engine needs, and it gives the test suite an exact answer to
//! measure the simulation against on the same side of the FFI boundary.

use crate::normal;

/// Call or put.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionRight {
    /// A call: the right to buy.
    Call,
    /// A put: the right to sell.
    Put,
}

impl OptionRight {
    /// Return +1 for a call and -1 for a put.
    #[inline]
    #[must_use]
    pub fn sign(self) -> f64 {
        match self {
            OptionRight::Call => 1.0,
            OptionRight::Put => -1.0,
        }
    }

    /// Build from the single-character market convention.
    ///
    /// # Errors
    /// Returns an error for anything other than `C` or `P`, in either case.
    pub fn from_char(c: char) -> Result<Self, String> {
        match c.to_ascii_uppercase() {
            'C' => Ok(OptionRight::Call),
            'P' => Ok(OptionRight::Put),
            other => Err(format!("option right must be C or P, got {other:?}")),
        }
    }
}

/// The undiscounted payoff of a European option at expiry.
#[inline]
#[must_use]
pub fn payoff(terminal: f64, strike: f64, right: OptionRight) -> f64 {
    (right.sign() * (terminal - strike)).max(0.0)
}

/// Price a European option off the forward.
#[must_use]
pub fn price(
    forward: f64,
    strike: f64,
    vol: f64,
    time: f64,
    discount: f64,
    right: OptionRight,
) -> f64 {
    let total_vol = vol * time.sqrt();
    if total_vol <= 0.0 || time <= 0.0 {
        return discount * payoff(forward, strike, right);
    }
    let d1 = ((forward / strike).ln() + 0.5 * total_vol * total_vol) / total_vol;
    let d2 = d1 - total_vol;
    let sign = right.sign();
    sign * discount * (forward * normal::cdf(sign * d1) - strike * normal::cdf(sign * d2))
}

/// Forward delta: the sensitivity of the price to the forward.
#[must_use]
pub fn delta(
    forward: f64,
    strike: f64,
    vol: f64,
    time: f64,
    discount: f64,
    right: OptionRight,
) -> f64 {
    let total_vol = vol * time.sqrt();
    if total_vol <= 0.0 || time <= 0.0 {
        let intrinsic = right.sign() * (forward - strike) > 0.0;
        return if intrinsic {
            discount * right.sign()
        } else {
            0.0
        };
    }
    let d1 = ((forward / strike).ln() + 0.5 * total_vol * total_vol) / total_vol;
    let sign = right.sign();
    sign * discount * normal::cdf(sign * d1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn put_call_parity_holds() {
        let (f, k, v, t, df) = (100.0, 90.0, 0.2, 1.5, 0.95);
        let call = price(f, k, v, t, df, OptionRight::Call);
        let put = price(f, k, v, t, df, OptionRight::Put);
        assert_abs_diff_eq!(call - put, df * (f - k), epsilon = 1e-12);
    }

    #[test]
    fn at_the_money_call_and_put_agree() {
        let value = price(100.0, 100.0, 0.25, 1.0, 0.97, OptionRight::Call);
        let other = price(100.0, 100.0, 0.25, 1.0, 0.97, OptionRight::Put);
        assert_abs_diff_eq!(value, other, epsilon = 1e-12);
    }

    #[test]
    fn expired_options_are_worth_intrinsic() {
        assert_abs_diff_eq!(
            price(110.0, 100.0, 0.2, 0.0, 1.0, OptionRight::Call),
            10.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            price(110.0, 100.0, 0.2, 0.0, 1.0, OptionRight::Put),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn delta_matches_a_finite_difference() {
        let (f, k, v, t, df) = (100.0, 95.0, 0.3, 0.75, 0.98);
        let h = 1e-5;
        let numerical = (price(f + h, k, v, t, df, OptionRight::Call)
            - price(f - h, k, v, t, df, OptionRight::Call))
            / (2.0 * h);
        assert_abs_diff_eq!(
            delta(f, k, v, t, df, OptionRight::Call),
            numerical,
            epsilon = 1e-7
        );
    }

    #[test]
    fn rights_parse_from_market_letters() {
        assert_eq!(OptionRight::from_char('c'), Ok(OptionRight::Call));
        assert_eq!(OptionRight::from_char('P'), Ok(OptionRight::Put));
        assert!(OptionRight::from_char('X').is_err());
    }
}
