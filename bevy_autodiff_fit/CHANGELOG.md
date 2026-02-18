# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-17

### Added

- 1D piecewise Chebyshev polynomial fitting:
  - `fit_dense` for uniformly sampled data (resample to Chebyshev nodes, DCT)
  - `fit_sparse` for scattered data (Householder QR least-squares)
  - `uniform_breakpoints` helper for evenly spaced segment boundaries
  - `ChebyshevSegment` with standalone Clenshaw evaluation
  - `PiecewiseFit` with automatic segment dispatch, `eval()` (clamping), and `try_eval()` (Result)
  - Standalone derivative evaluation via Chebyshev recurrence (`eval_derivative`)
- 2D tensor product Chebyshev fitting:
  - `fit_dense_2d` for rectangular grid data (separable per-axis DCT)
  - `fit_sparse_2d` for scattered (x, y, z) data (2D Chebyshev Vandermonde + QR)
  - `ChebyshevSegment2D` with nested Clenshaw evaluation
  - `PiecewiseFit2D` with rectangular segment grid dispatch
- AD graph integration:
  - `ChebyshevSegment::build_graph` builds 1D Clenshaw as AD nodes
  - `ChebyshevSegment2D::build_graph` builds nested 2D Clenshaw as AD nodes
  - `PiecewiseFit::build_segment_graph` for composing fits with other AD operations
  - Domain mapping included in graph — chain rule through the mapping is automatic
- Pre-compiled evaluation:
  - `PiecewiseCompiled` wraps `CompiledGraph`s for fast repeated 1D evaluation with derivatives
  - `PiecewiseCompiled2D` for fast repeated 2D evaluation with partial derivatives
- C^k continuity constraints:
  - `fit_sparse_continuous` enforces derivative matching at 1D segment boundaries
  - `ContinuityOptions` with configurable order and penalty weight
- Derivative reliability estimation:
  - `SegmentReliability` from Chebyshev coefficient decay (1D)
  - `SegmentReliability2D` with per-axis reliability (2D)
- Chebyshev math utilities:
  - Chebyshev nodes, coefficients (DCT), Clenshaw evaluation
  - Derivative coefficient recurrence
  - Linear interpolation for resampling

[0.1.0]: https://github.com/VisVivaSpace/bevy_autodiff/releases/tag/bevy_autodiff_fit-v0.1.0
