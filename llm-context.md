# bevy_autodiff

Automatic differentiation library using Bevy ECS as the computational graph backend.

## What This Crate Does

`bevy_autodiff` computes exact derivatives of mathematical functions by building a computation graph as ECS entities, then applying the chain rule symbolically. It supports:

- First-order gradients (reverse-mode, O(1) in number of inputs)
- Higher-order and mixed partial derivatives (forward-mode symbolic differentiation)
- Fast repeated evaluation via `CompiledGraph` (flat array, no ECS overhead)
- GPU batch evaluation via wgpu — evaluate at millions of input points in parallel (f32)
- 16 unary + 7 binary elementary operations (includes `PowLog`/`DivLog` logarithmic variants for f32-stable second-order derivatives)

## When to Use This Crate

Use `bevy_autodiff` when you need:
- Exact derivatives (not finite differences) of composed elementary functions
- Gradient computation for optimization, sensitivity analysis, or Jacobians
- Integration with a Bevy ECS application
- A simple, dependency-light AD library

## Installation

```toml
[dependencies]
bevy_autodiff = "0.8"
```

## Core API

Three ways to build computation graphs (see [Usage Guide](docs/usage_guide.md) for full details):

### `#[autodiff]` proc-macro (recommended)

```rust
use bevy_autodiff::{AutoDiff, autodiff};
use bevy_autodiff::ops::with_context;

#[autodiff]
fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0;
    let b = 100.0;
    (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
}

// Direct evaluation — works like a normal function
assert_eq!(rosenbrock(1.0, 1.0), 0.0);

// AD graph construction
let mut ad = AutoDiff::new();
let x = ad.var(1.0).unwrap();
let y = ad.var(1.0).unwrap();
let f = with_context(&mut ad, || rosenbrock(x, y));
```

Requires `features = ["proc-macros"]`. The macro makes the function generic over `T: DiffNum` — call with floats for direct evaluation, or with `Var` inside `with_context` for graph construction. Use `#[autodiff(stable_derivatives)]` to route `pow`/`div` through logarithmic variants for f32-stable second-order derivatives.

### `expr!` macro

```rust
use bevy_autodiff::{AutoDiff, expr};

let mut ad = AutoDiff::new();
let x = ad.var(2.0).unwrap();
let y = ad.var(3.0).unwrap();
let f = expr!(ad, x * x + x * y);
```

No feature required. Supports `+`, `-`, `*`, `/`, unary `-`, transcendental functions, and `pow`.

### Builder API

```rust
let mut ad = AutoDiff::new();
let x = ad.var(2.0).unwrap();
let y = ad.var(3.0).unwrap();
let c = ad.constant(5.0);
let f = ad.add(ad.mul(x, y), c);  // f = x*y + 5
```

### Reverse-mode gradient (recommended for first-order)

```rust
let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
cg.eval(&[2.0, 3.0]).unwrap();
let grad = cg.gradient().to_vec();  // [df/dx, df/dy] = [3.0, 2.0]

// Re-evaluate at new point without recompiling
cg.eval(&[4.0, 5.0]).unwrap();
let grad = cg.gradient().to_vec();  // [5.0, 4.0]
```

### Compiled higher-order derivatives

```rust
let mut cg = ad.compile_order(f, &[x, y], 2).unwrap();
cg.eval(&[2.0, 3.0]).unwrap();
let dfdx = cg.partial(&[1, 0]).unwrap();     // first partial w.r.t. x
let d2fdxdy = cg.partial(&[1, 1]).unwrap();  // mixed second partial
```

## Available Operations

All operations are methods on `AutoDiff`:

| Method | Math |
|--------|------|
| `add(a, b)` | a + b |
| `sub(a, b)` | a - b |
| `mul(a, b)` | a * b |
| `div(a, b)` | a / b |
| `pow(a, b)` | a^b |
| `neg(a)` | -a |
| `square(a)` | a^2 |
| `sqrt(a)` | sqrt(a) |
| `exp(a)` | e^a |
| `ln(a)` | ln(a) |
| `sin(a)`, `cos(a)`, `tan(a)` | trig |
| `asin(a)`, `acos(a)`, `atan(a)` | inverse trig |
| `sinh(a)`, `cosh(a)`, `tanh(a)` | hyperbolic |
| `asinh(a)`, `acosh(a)`, `atanh(a)` | inverse hyperbolic |
| `powi(a, n)` | a^n (integer) |
| `powf(a, p)` | a^p (float) |
| `pow_log(a, b)` | a^b (logarithmic differentiation) |
| `powi_log(a, n)` | a^n (logarithmic differentiation) |
| `powf_log(a, p)` | a^p (logarithmic differentiation) |
| `div_log(a, b)` | a/b (logarithmic differentiation) |

### GPU batch evaluation (requires `wgpu` feature)

```toml
bevy_autodiff = { version = "0.8", features = ["wgpu"] }
```

```rust
use bevy_autodiff::gpu::GpuContext;

let gpu = GpuContext::new()?;
let gpu_graph = gpu.prepare(&compiled_graph)?;
let results = gpu_graph.eval_batch(&gpu, &[&x_samples, &y_samples])?;
let values = results.values();           // &[f32]
let dfdx = results.partials(&[1, 0]);    // Option<&[f32]>
```

### WGSL code generation (no feature required)

```rust
let wgsl = compiled_graph.to_wgsl("my_func").unwrap();
// Returns a WGSL struct + function: embeddable in any shader
```

Generates a standalone WGSL function from a compiled graph. All 23 operations map to direct WGSL expressions (no interpreter loop). The output is a struct definition (`{FuncName}Output` with `value` + partial derivative fields) and a pure function. Does not require the `wgpu` feature — pure string generation.

## Key Types

- `AutoDiff<F>` -- the computation graph context, generic over float type (`f32`, `f64`, `Complex<f64>`, etc.). Wraps a Bevy ECS `World`.
- `Var` -- lightweight `Copy` handle to a graph entity (float-agnostic)
- `DiffNum` -- trait for dual-use functions: implemented for `f32`, `f64` (direct evaluation) and `Var` (graph construction)
- `CompiledGraph<F>` -- flattened graph for fast evaluation, gradient computation, and WGSL code generation. Derives `Clone`, `Component`, `Resource` for Bevy integration.
- `NodeOp<F>` -- single operation in the compiled flat array
- `GpuContext` -- holds wgpu device, queue, and compute pipeline (feature `wgpu`). Derives `Resource`.
- `GpuGraph` -- prepared GPU buffers for a compiled graph (feature `wgpu`). Derives `Component`, `Resource`.
- `GpuResults` -- GPU evaluation results with values and partials (feature `wgpu`)

## Limitations

- Maximum 64 input variables (dependency tracking uses a `u64` bitmask)
- `eval()` on `AutoDiff` returns construction-time values; use `CompiledGraph::eval()` for re-evaluation at new points
- `pow(a, b)` requires `a > 0` when `b` is non-integer
- GPU path uses f32 precision (CPU path uses f64)
