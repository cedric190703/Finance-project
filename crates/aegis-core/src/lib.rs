//! Numerical kernels for the Aegis risk engine.
//!
//! This crate is deliberately free of any Python dependency: it is a plain Rust
//! library that can be tested, benchmarked and reused on its own. The `PyO3`
//! bindings live in the sibling `aegis-py` crate.
//!
//! What lives here is the work that is too slow to do in Python and too
//! repetitive to do cleverly: normal distribution functions, Black's formula,
//! a low-discrepancy sequence, and the Monte Carlo engine that puts them
//! together.

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
// Pedantic clippy is on, with four exemptions that are properties of the domain
// rather than defects to fix:
#![allow(clippy::must_use_candidate)]
// Sample counts become divisors and path counts become denominators; converting
// a usize to f64 is what statistics on a sample is made of. Nothing here will
// see 2^53 paths.
#![allow(clippy::cast_precision_loss)]
// Finance is written in single letters. Renaming `k` to `strike_price` inside a
// formula makes it harder to check against the paper it came from, not easier.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

pub mod black;
pub mod mc;
pub mod normal;
pub mod sobol;

/// Version of the kernel library, surfaced through the Python bindings so a
/// running service can report exactly which compiled core it loaded.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_populated() {
        assert!(!VERSION.is_empty());
    }
}
