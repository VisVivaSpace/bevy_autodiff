# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-02-17

### Added

- Generic float types: `AutoDiff<F>`, `CompiledGraph<F>`, `Value<F>`, `NodeOp<F>` are now generic over `F: Float` — supports f32, f64, and user-defined numeric types
- `DiffNum` trait for dual-use functions: write `fn f<T: DiffNum>(x: T) -> T` and call with plain floats (direct evaluation) or `Var` (graph construction)
- `Float` trait for graph-storable types: extends `DiffNum` with `to_f64()` and `is_finite()`. User-extensible — implement for custom numeric types (interval arithmetic, extended precision, etc.)
- `DiffNum` implementations for `f32`, `f64`, and `Var`
- `Float` implementations for `f32` and `f64`
- `AutoDiff::<f32>::new()` for single-precision computation graphs
- New f32 test suite: 20 tests covering arithmetic, derivatives, CompiledGraph, gradient
- New DiffNum test suite: 21 tests covering generic function evaluation and differentiation
- Proc-macro tests for direct f64/f32 evaluation of `#[autodiff]` functions
- Proc-macro edge case tests: `i32` parameters, f64/f32 consistency, stable_derivatives div_log
- `Display` impls for `UnaryOp` and `BinaryOp` (delegates to `name()`)
- `PartialEq` derive on `AutoDiffError` (enables `assert_eq!` on errors)
- `Ord` and `PartialOrd` impls on `Var` (entity ordering, BTreeMap usage)
- `ValidationError` variant on `AutoDiffError` for graph validation
- `CompiledGraph::nodes()` is now public (inspect compiled graph for debugging)
- GPU tests for `DivLog` and `PowLog` op code mapping
- Additional constant folding in differentiation: `a - a → 0`, `a / a → 1`, `-1 * x → neg(x)`

### Changed

- **BREAKING:** `#[autodiff]` now generates `fn f<T: DiffNum>(x: T) -> T` instead of injecting `ad: &mut AutoDiff`. Call via `with_context(&mut ad, || f(x))` for graph construction, or `f(2.0_f64)` for direct evaluation.
- **BREAKING:** `AutoDiff`, `CompiledGraph`, `Value`, `NodeOp` are now generic over float type (existing code using `AutoDiff::new()` with f64 literals continues to work via type inference)
- **BREAKING:** Internal modules (`codegen`, `compiled`, `components`, `context`, `debug`, `diff_num`, `error`, `graph`, `var`) are now `pub(crate)` — use re-exports at crate root instead (e.g., `bevy_autodiff::CompiledGraph`, not `bevy_autodiff::compiled::CompiledGraph`)
- `Float` trait now requires `Display` (both `f32` and `f64` already implement it)
- `validate_graph` returns `Result<(), AutoDiffError>` instead of `Result<(), String>`
- `value()`, `partial()`, `gradient()` use `assert!` instead of `debug_assert!` for eval-before-use checks (catches misuse in release builds)
- `get_value` in graph traversal is now generic over `F: Float`
- docs.rs: added `#![cfg_attr(docsrs, feature(doc_cfg))]` for proper feature-gated documentation

### Fixed

- `pow` and `pow_log` primal evaluation now uses `powf` instead of `exp(y * ln(x))` — correctly handles negative bases (e.g., `(-2)^2 = 4` instead of `NaN`)
- `compile()` returns `Result` with `MultiIndexLengthMismatch` instead of panicking on multi-index length mismatch

### Removed

- Removed `num-complex` and `num-traits` dependencies — replaced with custom `Float` trait (user-extensible, no sealed trait limitation)

## [0.7.0] - 2026-02-16

### Added

- Bevy integration: `CompiledGraph` derives `Clone`, `Component`, and `Resource` for direct use in Bevy applications
- `GpuContext` derives `Resource`, `GpuGraph` derives `Component` and `Resource`
- `bevy_par_eval` example demonstrating parallel evaluation via `par_iter_mut()`
- Send+Sync compile-time assertions for `CompiledGraph`, `GpuContext`, and `GpuGraph`
- `MultiIndexLengthMismatch` error variant for `partial()` validation
- `Clone` derive on `AutoDiffError`
- `PartialEq` derive on `NodeOp`
- `Display` impl on `Var`
- Custom `Debug` impl on `CompiledGraph` (shows node count, not all data)
- `#[must_use]` on all graph-building methods
- `debug_assert!` on `value()`/`partial()`/`gradient()` if called before `eval()`
- `powi`, `powf`, `powi_log`, `powf_log` support in `expr!` macro
- "Why bevy_autodiff?" section in README

### Fixed

- `expr!` macro: `a - b - c` now correctly left-associates (was right-associative, giving `a - (b - c)`)
- `expr!` macro: `a / b / c` now correctly left-associates (was right-associative)
- `input_index()`, `set_inputs()`, `partial()` now return `Result` instead of panicking
- Proc-macro error handling: `panic!()` replaced with `syn::Error` for proper compile errors
- `validate_graph()` no longer panics on cycle detection (returns `Result`)
- Validation assertions added to `orbital_mechanics` and `gpu_batch` examples
- Fixed missing `.unwrap()` calls in README and usage guide code examples
- Fixed "Taylor coefficients" terminology in usage guide

### Changed

- **BREAKING:** Reduced public API surface — removed re-exports of internal types (`BinaryInputs`, `UnaryInput`, graph traversal helpers, optimizer utilities, free functions from `ops`)
- **BREAKING:** Added `#[non_exhaustive]` to `UnaryOp`, `BinaryOp`, and `NodeOp`
- **BREAKING:** `input_index()`, `set_inputs()`, `partial()` return `Result` instead of panicking
- `Dependencies.mask` is now `pub(crate)`
- `generate_multi_indices` is now `pub(crate)`
- Removed unused `optimize` and `util` modules (dead code)
- `cargo fmt` applied across entire codebase

## [0.6.0] - 2026-02-15

### Added

- Logarithmic derivative operations for f32-stable second-order derivatives:
  - `pow_log(base, exp)` / `powi_log(x, n)` / `powf_log(x, p)` — power operations that differentiate via `a^b · b · (da/a)` instead of `b · a^(b-1) · da`, avoiding catastrophic cancellation at second order
  - `div_log(a, b)` — division that differentiates via `(a/b) · (da/a - db/b)` instead of `(da·b - a·db) / b²`
  - `#[autodiff(stable_derivatives)]` attribute — automatically routes `pow`→`pow_log`, `powi`→`powi_log`, `powf`→`powf_log`, `/`→`div_log`
  - `expr!` macro support for `pow_log()` and `div_log()`
  - Free functions: `pow_log`, `powi_log`, `powf_log`, `div_log` for `with_context` usage

### Fixed

- GPU unit tests (`gpu/graph.rs`) now compile after `var()` and `compile_*()` API changed to return `Result`

### Changed

- Tightened test tolerances per aerospace numerical methods review (primal matches now use `assert_eq!`, first-order `1e-12`, second-order `1e-11`)

## [0.5.0] - 2026-02-15

### Added

- WGSL code generation: `CompiledGraph::to_wgsl(func_name)` emits a standalone WGSL struct + function from any compiled graph. No `wgpu` dependency required — pure string generation, embeddable in custom compute or fragment shaders.

### Changed

- Updated `wgpu` dependency from 27 to 28
- Updated MSRV from 1.89 to 1.92

## [0.4.0] - 2026-02-14

### Added

- GPU batch evaluation via wgpu (`wgpu` feature flag):
  - `GpuContext` for device acquisition (auto-select or from existing wgpu device)
  - `GpuGraph` for prepared per-graph GPU buffers, reusable across dispatches
  - `GpuResults` with `.values()` and `.partials()` for reading back results
  - Interpreter-style WGSL compute shader — zero warp divergence
  - SoA memory layout for coalesced GPU access
  - All 21 operations supported (16 unary + 5 binary)
  - Forward-mode symbolic partials on GPU
  - f32 precision on GPU (CPU path remains f64)
- `GpuError` error type with `#[non_exhaustive]` for future extensibility
- `gpu_batch` example demonstrating 1M-sample parallel evaluation
- GPU oracle test suite: 27 tests comparing GPU f32 against CPU f64
- 15 GPU unit tests covering dispatch, error paths, and graph reuse
- `docs.rs` feature badge for GPU module via `doc(cfg)` annotation

### Changed

- `CompiledGraph` gained `pub(crate)` accessors (`nodes()`, `output_index()`, `partial_outputs()`) gated behind `#[cfg(feature = "wgpu")]`

## [0.3.0] - 2026-02-13

### Changed

- Updated to Rust 2024 edition
- Updated `bevy_ecs` dependency from 0.15 to 0.18
- Updated `bevy_entity_ptr` dependency from 0.1 to 0.5
- Updated MSRV from 1.77 to 1.89
- Applied Rust 2024 `if let` chain idioms (collapsible `if` statements)
- Replaced removed `Entity::from_raw()` with `Entity::from_raw_u32()` in tests

## [0.2.0] - 2026-02-13

### Added

- Reverse-mode gradient computation on `CompiledGraph`:
  - `gradient()` computes all partial derivatives in a single backward pass
  - `gradient_of(output_node)` computes gradient from an arbitrary node
  - `eval_gradient(inputs)` convenience method for combined eval + gradient
- `compile_primal(output, inputs)` on `AutoDiff` for compiling without symbolic derivatives
- Adjoint helper functions (`unary_adjoint`, `binary_adjoint`) for all 21 operations
- Oracle comparison tests for `asinh`, `acosh`, `atanh` against the `autodiff` crate
- `reverse_gradient` example demonstrating gradient descent with reverse-mode
- `docs/architecture.md` documenting ECS graph representation and compilation pipeline
- `docs/numerical_precision.md` documenting precision tiers and tolerance justification
- `examples/README.md` describing all 7 examples

### Changed

- Cached constant `0.0` and `1.0` entities in `AutoDiff` to reduce entity growth during repeated differentiation
- Updated `README.md` to accurately describe the symbolic graph differentiation architecture
- Updated crate-level and `CompiledGraph` documentation

### Removed

- Unused `AutoDiffError` / `AutoDiffResult` types and `thiserror` dependency

## [0.1.0] - 2026-01-15

### Added

- Initial release
- ECS-based computation graph using Bevy ECS
- Symbolic graph differentiation via chain rule with constant folding
- 16 unary operations: `neg`, `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`
- 5 binary operations: `add`, `sub`, `mul`, `div`, `pow`
- `CompiledGraph` for fast repeated evaluation
- `compile_order` for pre-compiled higher-order derivative subgraphs
- `expr!` declarative macro for natural mathematical syntax
- `#[autodiff]` proc-macro for transforming regular functions (behind `proc-macros` feature)
- Oracle validation against the `autodiff` crate
- Operator overloading via `with_context`

[0.8.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VisVivaSpace/bevy_autodiff/releases/tag/v0.1.0
