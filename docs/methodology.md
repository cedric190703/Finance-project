# Methodology

## Pricing and sensitivities

European options are priced with Black's forward formula,
`V = DF [F N(d1) - K N(d2)]` for calls (with the signs reversed for puts).
Volatility comes from an SVI surface calibrated in total variance. Bonds are
the discounted value of remaining coupon and principal cash flows, including
accrued interest in their dirty price.

Reported Greeks use desk units: delta is cash P&L for a 1% spot move, vega is
cash P&L for one volatility point, theta is daily, and DV01 is cash P&L for a
one-basis-point fall in rates.

## Risk

Historical VaR fully revalues today's frozen book under historical factor moves.
Parametric VaR applies a normal quantile to the covariance-weighted factor
exposure; the optional Cornish-Fisher adjustment uses realised portfolio skew
and excess kurtosis. Monte Carlo draws correlated factor moves and fully
revalues the book. VaR is reported as a positive loss; Expected Shortfall is
the average loss at or beyond its 97.5% tail threshold.

## Validation

For a VaR confidence `c`, exceptions are days where realised P&L is below
`−VaR`, so the expected exception rate is `1−c`. Kupiec's POF test compares the
observed exception rate with that probability. Christoffersen's independence
test compares exception transition probabilities; conditional coverage is the
sum of their likelihood-ratio statistics. The Basel traffic light maps a
250-day, 99% exception count to green (0–4), amber (5–9), or red (10+).

## P&L explain

The explain uses opening-market Greeks and reconciles actual mark-to-market
P&L into delta, gamma, vega, theta, carry, rates and FX. `unexplained` is the
residual `actual P&L − explained P&L`; it is intentionally retained as a
control metric rather than forced to zero.

## Assumptions and limits

Historical factor proxies are visibly labelled in the risk report. Ten-day VaR
uses square-root-of-time scaling, which can understate risk in autocorrelated
or stressed markets. The engine prices vanilla European equity options and
fixed-rate bonds; it does not claim to model exotics, XVA, liquidity risk, or
intraday positions.
