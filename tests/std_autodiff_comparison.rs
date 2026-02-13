//! Comparison tests against std::autodiff
//!
//! Temporarily disabled during Taylor->symbolic differentiation migration.
//! Will be re-enabled in Step 5 after differentiate() is implemented.
//!
//! These tests require the Enzyme toolchain and are gated behind the
//! `std_autodiff_tests` feature flag, so they won't affect normal builds.

#![cfg(feature = "std_autodiff_tests")]
