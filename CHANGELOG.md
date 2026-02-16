# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Bevy integration: `CompiledGraph` derives `Clone`, `Component`, and `Resource` for direct use in Bevy applications
- `GpuContext` derives `Resource`, `GpuGraph` derives `Component` and `Resource`
- `bevy_par_eval` example demonstrating parallel evaluation via `par_iter_mut()`
- Send+Sync compile-time assertions for `CompiledGraph`, `GpuContext`, and `GpuGraph`

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

[0.6.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VisVivaSpace/bevy_autodiff/releases/tag/v0.1.0
