//! Numerical kernels for the Aegis risk engine.
//!
//! This crate is deliberately free of any Python dependency: it is a plain Rust
//! library that can be tested, benchmarked and reused on its own. The PyO3
//! bindings live in the sibling `aegis-py` crate.

#![warn(missing_docs)]

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
