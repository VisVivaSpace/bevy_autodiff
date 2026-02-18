# bevy_autodiff_fit

Piecewise Chebyshev polynomial fitting with exact derivative support via [`bevy_autodiff`](https://crates.io/crates/bevy_autodiff).

Fit smooth piecewise Chebyshev polynomials to tabulated data (1D or 2D), then differentiate them exactly through the autodiff graph. Derivatives are computed via the chain rule — the Clenshaw evaluation is built as graph nodes, so `differentiate()` just works.

## Features

- **1D and 2D tensor product** Chebyshev fitting
- **Dense fitting** (uniformly sampled data → DCT) and **sparse fitting** (scattered data → Householder QR)
- **Exact derivatives** via `bevy_autodiff` graph integration — Clenshaw evaluation as AD nodes
- **Compiled evaluation** (`PiecewiseCompiled`, `PiecewiseCompiled2D`) for fast repeated eval with derivatives
- **C^k continuity constraints** at segment boundaries via penalty method
- **Derivative reliability estimation** from Chebyshev coefficient decay
- **Standalone derivative evaluation** via Chebyshev recurrence (no AD graph needed)

## Installation

```toml
[dependencies]
bevy_autodiff_fit = "0.1"
```

## Quick Start

```rust
use bevy_autodiff_fit::{fit_dense, FitOptions, PiecewiseCompiled};

// Your data
let x: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
let y: Vec<f64> = x.iter().map(|&x| x.sin()).collect();

// Fit with a single segment, degree 20
let result = fit_dense(&x, &y, &[0.0, 1.0], &FitOptions { degree: 20 }).unwrap();

// Check reliability
println!("reliable up to order {}", result.reliability[0].max_reliable_order);

// Compile for fast evaluation with first derivatives
let mut compiled = PiecewiseCompiled::new(&result.fit, 1).unwrap();
compiled.eval(0.5).unwrap();
let value = compiled.value();
let derivative = compiled.partial(&[1]).unwrap();
```

## 2D Tensor Product Fitting

```rust
use bevy_autodiff_fit::{fit_dense_2d, FitOptions2D, PiecewiseCompiled2D};

let x_data: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
let y_data: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
let z_data: Vec<Vec<f64>> = y_data.iter()
    .map(|&y| x_data.iter().map(|&x| (x * y).sin()).collect())
    .collect();

let result = fit_dense_2d(
    &x_data, &y_data, &z_data,
    &[0.0, 1.0], &[0.0, 1.0],
    &FitOptions2D { degree_x: 10, degree_y: 10 },
).unwrap();

let mut compiled = PiecewiseCompiled2D::new(&result.fit, 1).unwrap();
compiled.eval(0.5, 0.5).unwrap();
let dfdx = compiled.partial(&[1, 0]).unwrap();
let dfdy = compiled.partial(&[0, 1]).unwrap();
```

## Integration with bevy_autodiff

Build Clenshaw evaluation as AD graph nodes for composition with other operations:

```rust
use bevy_autodiff::AutoDiff;
use bevy_autodiff_fit::{fit_dense, FitOptions};

let x_data: Vec<f64> = (0..=50).map(|i| i as f64 / 50.0).collect();
let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
let result = fit_dense(&x_data, &y_data, &[0.0, 1.0], &FitOptions { degree: 10 }).unwrap();

let mut ad = AutoDiff::new();
let x = ad.var(0.5).unwrap();

// Build Clenshaw graph for segment 0
let f = result.fit.build_segment_graph(&mut ad, x, 0).unwrap();

// Compose with other operations
let g = ad.exp(f);  // g = exp(fit(x))

// Differentiate — chain rule is automatic
let dg = ad.differentiate(g, x).unwrap();
```

## Important

The fit is an **approximation**, not exact. Derivatives of the fit approximate derivatives of the underlying function, with accuracy limited by:
- Polynomial degree (higher = more accurate, but amplifies noise)
- Data quality (noisy data → unreliable higher derivatives)
- Segment width (narrower segments → less polynomial bending)

Use `FitResult::reliability` to check how many derivatives are trustworthy.

## Documentation

- [API Reference](https://docs.rs/bevy_autodiff_fit)

## Development

This project was co-developed with [Claude](https://claude.ai), an AI assistant by Anthropic.

## License

MIT
