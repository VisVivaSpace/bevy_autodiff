# bevy_autodiff

Automatic differentiation using [Bevy ECS](https://bevyengine.org/) as the computational graph backend.

Write normal Rust functions, get exact derivatives — including gradients, Jacobians, and Hessians. GPU batch evaluation, WGSL code generation, and native Bevy ECS integration included.

## Why bevy_autodiff?

**Arbitrary-order exact derivatives.** Most automatic differentiation libraries provide first-order gradients. `bevy_autodiff` computes exact partial derivatives to any order — second-order Hessians, third-order tensors, mixed partials — via successive symbolic differentiation. No finite differences, no truncation.

**GPU batch evaluation.** Evaluate a compiled function and all its derivatives at millions of input points in parallel on the GPU. Generate standalone WGSL shader functions from any compiled graph. As far as we know, no other Rust AD crate offers either capability.

**Data-oriented design.** Most AD libraries represent computation graphs as opaque internal structures with behavior baked in. `bevy_autodiff` takes a different approach: the graph *is* ECS data. Variables are entities, operations are components, and differentiation is a pure traversal that reads the graph and writes new entities. Data is separated from code at every level — a direct application of data-oriented programming to automatic differentiation. The compiled output (`CompiledGraph`) is a flat, cache-friendly array with no entity references or indirection.

This design makes the graph open and extensible: adding new metadata to nodes is just adding a new component. It also means `CompiledGraph` is a Bevy `Component` and `Resource` out of the box — compiled graphs live directly in your app's ECS, evaluated in parallel with `par_iter_mut()`.

## Features

- **Write normal Rust functions, get exact derivatives** — `#[autodiff]` makes functions dual-use: evaluate directly with floats or build AD graphs with `Var`
- **Generic float types** — `AutoDiff<f32>`, `AutoDiff<f64>`, or any type implementing the `Float` trait
- **GPU batch evaluation** — evaluate at millions of input points in parallel via wgpu (Metal, Vulkan, DX12)
- **WGSL code generation** — embed derivative functions directly in any shader
- **Bevy-native** — `CompiledGraph` is a `Component` + `Resource`; use `par_iter_mut()` for parallel evaluation
- **Reverse-mode gradient** — full gradient in a single backward pass, O(1) in input count
- **Higher-order exact derivatives** — Hessians, mixed partials via successive symbolic differentiation
- **f32-stable second-order** — `#[autodiff(stable_derivatives)]` routes through logarithmic derivative variants
- **23 elementary operations** — all with symbolic differentiation rules and reverse-mode adjoints

## Installation

```toml
[dependencies]
bevy_autodiff = { version = "0.8", features = ["proc-macros"] }
```

Without proc-macros (uses `expr!` macro or builder API only):

```toml
[dependencies]
bevy_autodiff = "0.8"
```

## Quick Start

The `#[autodiff]` attribute transforms regular Rust functions into dual-use functions that work with both plain floats (direct evaluation) and `Var` (graph construction for automatic differentiation):

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

// AD graph construction — same function, called with Var
let mut ad = AutoDiff::new();
let x = ad.var(1.0).unwrap();
let y = ad.var(1.0).unwrap();
let f = with_context(&mut ad, || rosenbrock(x, y));

// Reverse-mode gradient (recommended for first-order)
let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
cg.eval(&[0.5, 0.8]).unwrap();
let grad = cg.gradient();  // [df/dx, df/dy]

// Re-evaluate at a new point without recompiling
cg.eval(&[1.0, 1.0]).unwrap();
let grad = cg.gradient();  // [0.0, 0.0] — the minimum
```

Under the hood, `#[autodiff]` makes the function generic over `T: DiffNum`. The `DiffNum` trait is implemented for `f32`, `f64` (direct evaluation), and `Var` (graph construction).

For second-order derivatives, use `compile_order`:

```rust
let mut cg = ad.compile_order(f, &[x, y], 2).unwrap();
cg.eval(&[0.5, 0.8]).unwrap();

let dfdx   = cg.partial(&[1, 0]).unwrap();  // df/dx
let dfdy   = cg.partial(&[0, 1]).unwrap();  // df/dy
let d2fdx2 = cg.partial(&[2, 0]).unwrap();  // d²f/dx²
let d2fdy2 = cg.partial(&[0, 2]).unwrap();  // d²f/dy²
let d2mix  = cg.partial(&[1, 1]).unwrap();  // d²f/dxdy
```

The `stable_derivatives` attribute automatically routes power and division operations to their logarithmic variants, which avoid catastrophic cancellation in f32 second-order derivatives:

```rust
#[autodiff(stable_derivatives)]
fn gravity(r2: f64) -> f64 {
    // pow and / are automatically routed to pow_log and div_log
    r2.powf(-1.5) * r2  // uses powf_log internally
}
```

## Expression Macro

The `expr!` macro provides natural mathematical syntax without requiring the `proc-macros` feature:

```rust
use bevy_autodiff::{AutoDiff, expr};

let mut ad = AutoDiff::new();
let x = ad.var(2.0).unwrap();
let y = ad.var(3.0).unwrap();

let f = expr!(ad, x * x + x * y);
assert_eq!(ad.eval(f).unwrap(), 10.0); // 4 + 6
```

See the [Usage Guide](docs/usage_guide.md) for a comparison of all three API tiers (`#[autodiff]`, `expr!`, and the builder API).

## GPU Batch Evaluation

Evaluate compiled graphs at millions of input points in parallel on the GPU. Useful for Monte Carlo simulation, batch trajectory optimization, or any workload that evaluates the same function at many different inputs.

```toml
[dependencies]
bevy_autodiff = { version = "0.8", features = ["wgpu"] }
```

```rust
use bevy_autodiff::AutoDiff;
use bevy_autodiff::gpu::GpuContext;

let gpu = GpuContext::new().unwrap();

let mut ad = AutoDiff::new();
let x = ad.var(0.0).unwrap();
let y = ad.var(0.0).unwrap();
let xy = ad.mul(x, y);
let f = ad.add(ad.sin(xy), ad.exp(x));

let graph = ad.compile_order(f, &[x, y], 1).unwrap();
let gpu_graph = gpu.prepare(&graph).unwrap();

// Evaluate at 1M points in parallel
let x_samples: Vec<f32> = (0..1_000_000).map(|i| i as f32 * 1e-6).collect();
let y_samples: Vec<f32> = (0..1_000_000).map(|i| i as f32 * 1e-6).collect();
let results = gpu_graph.eval_batch(&gpu, &[&x_samples, &y_samples]).unwrap();

let values = results.values();          // f(x,y) for each sample
let dfdx = results.partials(&[1, 0]);   // df/dx for each sample
```

The GPU path uses f32 precision (the CPU path uses f64). A WGSL interpreter kernel dispatches one GPU thread per sample with zero warp divergence.

## WGSL Code Generation

Generate standalone WGSL functions from compiled graphs — no `wgpu` dependency required. The output is a struct + function that can be embedded in any WGSL shader (custom compute kernels, fragment shaders, procedural generation).

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::new();
let x = ad.var(0.0).unwrap();
let y = ad.var(0.0).unwrap();
let f = ad.add(ad.sin(ad.mul(x, y)), ad.exp(x));
let graph = ad.compile_order(f, &[x, y], 1).unwrap();

let wgsl = graph.to_wgsl("my_func").unwrap();
```

This emits a self-contained WGSL snippet with a result struct and a pure function using direct WGSL expressions (no interpreter loop):

```wgsl
struct MyFuncOutput {
    value: f32,
    d1_0: f32,   // df/dx
    d0_1: f32,   // df/dy
}

fn my_func(p0: f32, p1: f32) -> MyFuncOutput {
    let v0 = p0;
    let v1 = p1;
    let v2 = v0 * v1;
    let v3 = sin(v2);
    // ... derivative nodes ...
    return MyFuncOutput(v3, ...);
}
```

Complements the interpreter-based GPU dispatch: the interpreter is a self-contained "eval at N points" path, while codegen produces an embeddable function for use inside other shaders.

## Bevy Integration

`CompiledGraph` derives `Clone`, `Component`, and `Resource`, so compiled graphs can live directly in your Bevy app's ECS. Use `par_iter_mut()` to evaluate many graphs in parallel via Bevy's `ComputeTaskPool`.

```rust
use bevy::prelude::*;
use bevy_autodiff::{AutoDiff, CompiledGraph};

#[derive(Component)]
struct InputPoint { x: f64, y: f64 }

// Build once with AutoDiff, clone to many entities
let mut ad = AutoDiff::new();
let x = ad.var(0.0).unwrap();
let y = ad.var(0.0).unwrap();
let f = ad.add(ad.square(x), ad.mul(x, y));
let template = ad.compile_primal(f, &[x, y]).unwrap();

// Parallel evaluation system
fn eval_system(mut query: Query<(&InputPoint, &mut CompiledGraph)>) {
    query.par_iter_mut().for_each(|(point, mut cg)| {
        cg.eval(&[point.x, point.y]).unwrap();
    });
}
```

`AutoDiff` uses its own private `World` for graph construction. `CompiledGraph` is the compiled artifact that crosses into your application's ECS — a flat array with no entity references.

GPU types also integrate: `GpuContext` derives `Resource` (singleton), `GpuGraph` derives `Component` and `Resource`.

## Performance

`CompiledGraph` flattens the ECS graph into a plain array — evaluation is **~130x faster** than rebuilding through ECS. Reverse-mode computes the **full gradient in the same time** regardless of input count. A realistic 3x6 Jacobian (two-body gravity, 6 state variables) takes about **2x the time of a single row** — reverse-mode pays for itself with just 2 inputs.

GPU batch evaluation processes **millions of function+derivative evaluations per second** on consumer hardware.

Run `cargo bench` for performance numbers on your system.

## Supported Operations

| Category | Operations |
|----------|-----------|
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `square` |
| Powers | `sqrt`, `pow`, `powi`, `powf` |
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Exponential | `exp`, `ln` |
| Logarithmic derivatives | `pow_log`, `powi_log`, `powf_log`, `div_log` |

The logarithmic derivative variants (`pow_log`, `div_log`) produce identical primal values but use a different symbolic differentiation rule that avoids catastrophic cancellation in f32 at second order. Use them when computing Hessians or second-order partials that will be evaluated in f32 (e.g., on GPU), or use `#[autodiff(stable_derivatives)]` to route automatically. See [Numerical Precision](docs/numerical_precision.md) for details.

## Higher-Order Derivatives

`compile_order` pre-compiles symbolic derivative subgraphs up to the requested order. Useful for Hessians, mixed partials, and beyond:

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::new();
let x = ad.var(1.0).unwrap();
let y = ad.var(2.0).unwrap();

let xy = ad.mul(x, y);
let x2 = ad.square(x);
let f = ad.add(x2, xy);

// Compile with all partials up to order 2
let mut cg = ad.compile_order(f, &[x, y], 2).unwrap();
cg.eval(&[1.0, 2.0]).unwrap();

let dfdx  = cg.partial(&[1, 0]).unwrap();  // df/dx = 2x + y = 4
let dfdy  = cg.partial(&[0, 1]).unwrap();  // df/dy = x = 1
let d2fdx = cg.partial(&[2, 0]).unwrap();  // d²f/dx² = 2
let d2mix = cg.partial(&[1, 1]).unwrap();  // d²f/dxdy = 1
```

For first-order gradients, prefer `compile_primal` + `gradient()` (reverse-mode) — it computes all partial derivatives in a single backward pass, O(1) in the number of inputs.

## How It Works

`bevy_autodiff` builds a computation graph as ECS entities in a private Bevy `World`. `differentiate(output, wrt)` walks the graph in topological order and applies the chain rule at every node, creating new entities for the derivative subgraph. Constant folding eliminates zero and one terms during differentiation to prevent graph bloat.

`compile()` flattens the entire graph (primal + derivative subgraphs) into a `Vec<NodeOp>` — a topologically sorted flat array. A single forward pass evaluates all values. For reverse-mode, one additional backward pass propagates adjoints to compute the full gradient.

For more detail, see [Architecture](docs/architecture.md).

### ECS Findings

This project explores what ECS can do for automatic differentiation. ECS is used for graph construction and symbolic differentiation only — all hot paths (`eval()`, `gradient()`, GPU `eval_batch()`) operate on flat arrays with no entity lookups. ECS provides open extensibility (new metadata = new component), cache-friendly SoA layout, and natural Bevy integration. The main cost is `bevy_ecs` compile time, which is confined to the cold path.

See [Architecture — What Does ECS Actually Bring?](docs/architecture.md#what-does-ecs-actually-bring) for the full analysis.

## Builder API

For full control over graph construction, use the builder methods directly on `AutoDiff`:

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::new();
let x = ad.var(2.0).unwrap();

// Build computation graph: f(x) = x² + 3x + 1
let x_squared = ad.square(x);
let three = ad.constant(3.0);
let three_x = ad.mul(three, x);
let one = ad.constant(1.0);
let sum = ad.add(x_squared, three_x);
let f = ad.add(sum, one);

assert_eq!(ad.eval(f).unwrap(), 11.0); // f(2) = 4 + 6 + 1

// Symbolic differentiation
let dfdx = ad.differentiate(f, x).unwrap();
assert_eq!(ad.eval(dfdx).unwrap(), 7.0);  // f'(2) = 2·2 + 3

// Higher-order via successive differentiation
assert_eq!(ad.derivative(f, x, 2).unwrap(), 2.0);  // f''(x) = 2
assert_eq!(ad.derivative(f, x, 3).unwrap(), 0.0);  // f'''(x) = 0
```

## Examples

See [`examples/README.md`](examples/README.md) for descriptions. Run with:

```bash
cargo run --example basic              # Basic derivatives
cargo run --example gradient           # Forward-mode gradient
cargo run --example reverse_gradient   # Reverse-mode gradient + gradient descent
cargo run --example hessian            # Hessian via successive differentiation
cargo run --example rosenbrock         # Rosenbrock optimization
cargo run --example orbital_mechanics  # Gravitational potential derivatives
cargo run --example stm_propagation    # State transition matrix propagation
cargo run --example bevy_par_eval      # Parallel eval via Bevy ECS
cargo run --example gpu_batch --features wgpu  # GPU batch evaluation
```

The examples use the builder API for clarity. For ergonomic function definition, see `#[autodiff]` and `expr!` in the [Usage Guide](docs/usage_guide.md).

## Testing

```bash
cargo test                                          # Unit + oracle + doc tests
cargo test --features proc-macros                   # Proc-macro tests
cargo test --features wgpu                          # GPU tests (requires GPU)
cargo test --test autodiff_crate_comparison         # Oracle: autodiff crate
cargo test --test gpu_cpu_comparison --features wgpu # Oracle: GPU vs CPU
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme test \
  --features std_autodiff_tests                     # Oracle: Enzyme
```

The test suite validates correctness through:

| Test type | What it validates | Count |
|-----------|-------------------|-------|
| Unit tests | Graph construction, all 23 operations, derivative properties, constant folding, CompiledGraph eval, reverse-mode adjoint formulas, reverse-mode backward pass, WGSL codegen, f32 graphs, DiffNum trait, Bevy trait bounds | 352 |
| Proc-macro tests | `#[autodiff]`, `expr!` macro, `stable_derivatives` attribute, dual-use f64/f32 evaluation | 79 |
| GPU unit tests | NodeOp conversion, GPU dispatch, buffer readback, error paths, Bevy trait bounds | 16 |
| Oracle (autodiff crate) | First derivatives against independent forward-mode AD | 22 |
| Oracle (GPU vs CPU) | GPU f32 results against CPU f64 for all ops, compositions, partials, batch sizes | 27 |
| Doc-tests | Code examples in documentation | 16 |
| Cross-validation | Reverse-mode gradient matches forward-mode symbolic partials | 8 (within unit) |

## Documentation

- [Usage Guide](docs/usage_guide.md) -- three API tiers: `#[autodiff]`, `expr!`, builder
- [Architecture](docs/architecture.md) -- ECS graph representation, compilation pipeline, differentiation approaches
- [Numerical Precision](docs/numerical_precision.md) -- precision tiers, tolerance justification, known considerations
- [API Reference](https://docs.rs/bevy_autodiff) -- rustdoc on docs.rs

## Development

This project was co-developed with [Claude](https://claude.ai), an AI assistant by Anthropic.

## License

MIT
