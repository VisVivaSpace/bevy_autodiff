# bevy_autodiff

Automatic differentiation using [Bevy ECS](https://bevyengine.org/) as the computational graph backend.

Variables are ECS entities, operations are components, and derivatives are computed by symbolic graph differentiation with chain-rule constant folding. An exploration of what ECS can do for automatic differentiation.

## Features

- **ECS as computation graph** -- entities are variables, components define operations and connectivity
- **Symbolic graph differentiation** -- `differentiate(output, wrt)` creates new entities representing the derivative graph via the chain rule
- **Successive differentiation** -- higher-order and mixed partials by repeated differentiation: `d²f/dxdy = differentiate(differentiate(f, x), y)`
- **Constant folding** -- zero/one terms are eliminated during differentiation to prevent graph bloat
- **CompiledGraph** -- flattens the ECS graph into a `Vec<NodeOp>` for fast repeated evaluation without ECS overhead
- **Reverse-mode gradient** -- single backward pass over CompiledGraph computes the full gradient regardless of input count
- **Forward-mode symbolic partials** -- pre-compiled derivative subgraphs for higher-order derivatives
- **21 elementary operations** -- 16 unary + 5 binary, all with differentiation rules and reverse-mode adjoints
- **GPU batch evaluation** -- evaluate compiled graphs at millions of input points in parallel via wgpu (Metal, Vulkan, DX12)

## Installation

```toml
[dependencies]
bevy_autodiff = "0.4"
```

## Quick Start

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::new();

// Create input variable
let x = ad.var(2.0);

// Build computation graph: f(x) = x² + 3x + 1
let x_squared = ad.square(x);
let three = ad.constant(3.0);
let three_x = ad.mul(three, x);
let one = ad.constant(1.0);
let sum = ad.add(x_squared, three_x);
let f = ad.add(sum, one);

// Evaluate
assert_eq!(ad.eval(f), 11.0); // f(2) = 4 + 6 + 1

// Symbolic differentiation
let dfdx = ad.differentiate(f, x);
assert_eq!(ad.eval(dfdx), 7.0);  // f'(2) = 2·2 + 3

// Higher-order via successive differentiation
assert_eq!(ad.derivative(f, x, 2), 2.0);  // f''(x) = 2
assert_eq!(ad.derivative(f, x, 3), 0.0);  // f'''(x) = 0
```

## Gradients

### Reverse-mode (recommended for many inputs)

`compile_primal` compiles only the function value. `gradient()` computes all partial derivatives in a single backward pass -- O(1) in the number of inputs.

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::new();
let x = ad.var(1.0);
let y = ad.var(2.0);

// f(x, y) = x² + x·y + y²
let x2 = ad.square(x);
let xy = ad.mul(x, y);
let y2 = ad.square(y);
let sum = ad.add(x2, xy);
let f = ad.add(sum, y2);

let mut cg = ad.compile_primal(f, &[x, y]);
cg.eval(&[1.0, 2.0]);

let val = cg.value();                   // 7.0
let grad = cg.gradient();               // [4.0, 5.0]

// Re-evaluate at new point without recompiling
cg.eval(&[3.0, -1.0]);
let grad = cg.gradient();               // [5.0, 1.0]
```

### Forward-mode (supports higher-order)

`compile_order` pre-compiles symbolic derivative subgraphs. Useful when you need second-order or mixed partial derivatives.

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::new();
let x = ad.var(1.0);
let y = ad.var(2.0);

let xy = ad.mul(x, y);
let f = ad.add(ad.square(x), xy);

// Compile with all partials up to order 2
let mut cg = ad.compile_order(f, &[x, y], 2);
cg.eval(&[1.0, 2.0]);

let dfdx  = cg.partial(&[1, 0]);  // df/dx = 2x + y = 4
let dfdy  = cg.partial(&[0, 1]);  // df/dy = x = 1
let d2fdx = cg.partial(&[2, 0]);  // d²f/dx² = 2
let d2mix = cg.partial(&[1, 1]);  // d²f/dxdy = 1
```

## Supported Operations

| Category | Operations |
|----------|-----------|
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `square` |
| Powers | `sqrt`, `pow`, `powi`, `powf` |
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Exponential | `exp`, `ln` |

## Expression Macros

The `expr!` macro provides natural mathematical syntax:

```rust
use bevy_autodiff::{AutoDiff, expr};

let mut ad = AutoDiff::new();
let x = ad.var(2.0);
let y = ad.var(3.0);

let f = expr!(ad, x * x + x * y);
assert_eq!(ad.eval(f), 10.0); // 4 + 6
```

With the `proc-macros` feature, the `#[autodiff]` attribute transforms regular functions:

```toml
[dependencies]
bevy_autodiff = { version = "0.4", features = ["proc-macros"] }
```

```rust
use bevy_autodiff::{AutoDiff, Var, autodiff};

#[autodiff]
fn rosenbrock(x: Var, y: Var) -> Var {
    let a = 1.0;
    let b = 100.0;
    (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
}

let mut ad = AutoDiff::new();
let x = ad.var(1.0);
let y = ad.var(1.0);
let f = rosenbrock(&mut ad, x, y);
```

## GPU Batch Evaluation

Enable the `wgpu` feature to evaluate compiled graphs on the GPU at millions of input points in parallel. Useful for Monte Carlo simulation, batch trajectory optimization, or any workload that evaluates the same function at many different inputs.

```toml
[dependencies]
bevy_autodiff = { version = "0.4", features = ["wgpu"] }
```

```rust
use bevy_autodiff::AutoDiff;
use bevy_autodiff::gpu::GpuContext;

let gpu = GpuContext::new().unwrap();

let mut ad = AutoDiff::new();
let x = ad.var(0.0);
let y = ad.var(0.0);
let xy = ad.mul(x, y);
let f = ad.add(ad.sin(xy), ad.exp(x));

let graph = ad.compile_order(f, &[x, y], 1);
let gpu_graph = gpu.prepare(&graph).unwrap();

// Evaluate at 1M points in parallel
let x_samples: Vec<f32> = (0..1_000_000).map(|i| i as f32 * 1e-6).collect();
let y_samples: Vec<f32> = (0..1_000_000).map(|i| i as f32 * 1e-6).collect();
let results = gpu_graph.eval_batch(&gpu, &[&x_samples, &y_samples]).unwrap();

let values = results.values();          // f(x,y) for each sample
let dfdx = results.partials(&[1, 0]);   // df/dx for each sample
```

The GPU path compiles the graph to f32 (the CPU path uses f64). A WGSL interpreter kernel dispatches one GPU thread per sample with zero warp divergence.

## How It Works

### Symbolic Graph Differentiation

`differentiate(output, wrt)` walks the computation graph in topological order and applies the chain rule at every node, creating new ECS entities for the derivative subgraph:

1. **Topological sort** from `output` back to inputs
2. **Base cases**: `d(wrt)/d(wrt) = 1`, `d(other_input)/d(wrt) = 0`, `d(constant)/d(wrt) = 0`
3. **Chain rule** at each operation node creates derivative entities
4. **Constant folding** via `smart_add`, `smart_mul`, etc. collapses zero/one terms

For higher-order: `differentiate(differentiate(f, x), y)` gives d²f/dxdy.

### CompiledGraph

`compile()` flattens the ECS graph (and any pre-built derivative subgraphs) into a `Vec<NodeOp>` -- a topologically sorted array of simple operations. A single forward pass evaluates all values.

For first-order gradients, `compile_primal()` + `gradient()` uses reverse-mode: one forward pass caches values, then one backward pass propagates adjoints to compute all partial derivatives simultaneously.

### ECS Architecture

The Bevy ECS world stores the computation graph:
- **Entities** represent variables (inputs, constants, intermediate results)
- **Components** store values (`Value`), operations (`UnaryOp`, `BinaryOp`), connectivity (`UnaryInput`, `BinaryInputs`), and dependency bitmasks (`Dependencies`)

### What Does ECS Actually Bring?

This project is an exploration of ECS for automatic differentiation. After four releases, here is what we've found.

**Where ECS is used:** Graph construction and symbolic differentiation only. When you call `var()`, `mul()`, `differentiate()`, etc., entities and components are created in a Bevy `World`. Once you call `compile()`, the ECS graph is flattened into a `Vec<NodeOp>` and the World is no longer involved.

**Where ECS is not used:** All hot paths bypass ECS entirely. `eval()`, `gradient()`, and GPU `eval_batch()` operate on flat arrays with no entity lookups. This is by design -- ECS builds the graph, compiled arrays run it.

**What ECS provides:**
- **Open extensibility** -- adding new metadata to graph nodes (e.g., dependency bitmasks) is just adding a component. No core struct modifications required, and downstream code can attach arbitrary data to graph entities without touching the library.
- **Structure-of-Arrays by default** -- Bevy's archetypal storage is inherently SoA, giving cache-friendly data layout for graph traversal without manual layout work.
- **Parallel-ready** -- batch operations like differentiating w.r.t. multiple variables, CSE detection, or graph analysis passes could be parallelized via ECS systems. A `Vec<Node> + HashMap` approach would need manual threading.
- **Inspectability** -- the graph is queryable through standard ECS patterns. Debug visualization, validation passes, and the DOT exporter all work through component queries.
- **Bevy integration** -- if you're already in a Bevy application, the computation graph lives in a `World` you can inspect and extend naturally.

**What ECS costs:**
- `bevy_ecs` adds compile time (the heaviest dependency).
- Entity creation has more overhead than a vec push (archetype lookup, component insertion).
- `World` borrowing requires the extract-before-mutate pattern in `differentiate()`.

All three costs are confined to graph construction -- a cold path that runs once. The compile-time cost is the most noticeable in practice.

**Bottom line:** The algorithms that make this crate useful (symbolic differentiation, reverse-mode gradient, GPU dispatch) are independent of ECS. The architecture that makes it extensible and well-structured comes from ECS. The performance-critical paths compile out of ECS completely, so there is no runtime cost where it matters.

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
cargo run --example gpu_batch --features wgpu  # GPU batch evaluation
```

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

The test suite (339 tests with `--features wgpu`) validates correctness through:

| Test type | What it validates | Count |
|-----------|-------------------|-------|
| Unit tests | Graph construction, all 21 operations, derivative properties, constant folding, CompiledGraph eval, reverse-mode adjoint formulas, reverse-mode backward pass | 261 |
| GPU unit tests | NodeOp conversion, GPU dispatch, buffer readback, error paths | 15 |
| Oracle (autodiff crate) | First derivatives against independent forward-mode AD | 22 |
| Oracle (GPU vs CPU) | GPU f32 results against CPU f64 for all ops, compositions, partials, batch sizes | 27 |
| Doc-tests | Code examples in documentation | 14 |
| Cross-validation | Reverse-mode gradient matches forward-mode symbolic partials | 8 (within unit) |

## Documentation

- [Architecture](docs/architecture.md) -- ECS graph representation, compilation pipeline, differentiation approaches
- [Numerical Precision](docs/numerical_precision.md) -- precision tiers, tolerance justification, known considerations
- [API Reference](https://docs.rs/bevy_autodiff) -- rustdoc on docs.rs

## Development

This project was co-developed with [Claude](https://claude.ai), an AI assistant by Anthropic.

## License

MIT
