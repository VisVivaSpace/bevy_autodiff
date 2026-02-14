# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/VisVivaSpace/bevy_autodiff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VisVivaSpace/bevy_autodiff/releases/tag/v0.1.0
