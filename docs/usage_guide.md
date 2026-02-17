# Usage Guide

`bevy_autodiff` provides three ways to build computation graphs, each offering a different balance of ergonomics and control. This guide covers all three and helps you choose.

## API Tiers

| Tier | API | Feature Required | Best For |
|------|-----|-----------------|----------|
| 1 | `#[autodiff]` proc-macro | `proc-macros` | Most use cases — write normal Rust functions |
| 2 | `expr!` declarative macro | None | Inline one-off expressions |
| 3 | Builder API | None | Dynamic graph construction, programmatic generation |

## Tier 1: `#[autodiff]` Proc-Macro (Recommended)

The `#[autodiff]` attribute transforms regular Rust functions into dual-use functions generic over `T: DiffNum`. Call with plain floats for direct evaluation, or with `Var` inside `with_context` for AD graph construction.

### Setup

```toml
[dependencies]
bevy_autodiff = { version = "0.8", features = ["proc-macros"] }
```

### Basic Usage

```rust
use bevy_autodiff::{AutoDiff, autodiff};
use bevy_autodiff::ops::with_context;

#[autodiff]
fn quadratic(x: f64) -> f64 {
    x * x + 2.0 * x + 1.0
}

// Direct evaluation — works like a normal function
assert_eq!(quadratic(2.0_f64), 9.0);
assert_eq!(quadratic(2.0_f32), 9.0_f32);  // f32 works too

// AD graph construction
let mut ad = AutoDiff::new();
let x = ad.var(2.0).unwrap();
let f = with_context(&mut ad, || quadratic(x));

assert_eq!(ad.eval(f).unwrap(), 9.0);  // f(2) = 4 + 4 + 1
```

The macro makes the function generic over `T: DiffNum`. Float literals become `T::from_f64(value)`, operators use `DiffNum` supertraits, and method calls like `.sin()` dispatch through the `DiffNum` trait.

### What the Macro Transforms

| You write | Macro generates |
|-----------|----------------|
| `x + y` | `x + y` (via `DiffNum: Add`) |
| `x - y` | `x - y` (via `DiffNum: Sub`) |
| `x * y` | `x * y` (via `DiffNum: Mul`) |
| `x / y` | `x / y` (via `DiffNum: Div`) |
| `-x` | `-x` (via `DiffNum: Neg`) |
| `x.sin()` | `DiffNum::sin(x)` |
| `x.cos()` | `DiffNum::cos(x)` |
| `x.exp()` | `DiffNum::exp(x)` |
| `x.ln()` | `DiffNum::ln(x)` |
| `x.sqrt()` | `DiffNum::sqrt(x)` |
| `x.square()` | `DiffNum::square(x)` |
| `x.powf(y)` | `DiffNum::powf(x, y)` |
| `x.powi(n)` | `DiffNum::powi(x, n)` |
| `3.14` | `T::from_f64(3.14)` |
| `f64`/`f32`/`Var` params | `T` |

All trig, hyperbolic, and inverse functions are supported. Free function calls like `sin(x)` are also transformed to method syntax.

### Multivariate Functions

```rust
use bevy_autodiff::ops::with_context;

#[autodiff]
fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0;
    let b = 100.0;
    (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
}

// Direct evaluation
assert_eq!(rosenbrock(1.0, 1.0), 0.0);

// AD graph construction
let mut ad = AutoDiff::new();
let x = ad.var(1.0).unwrap();
let y = ad.var(1.0).unwrap();
let f = with_context(&mut ad, || rosenbrock(x, y));

// Compile and evaluate
let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
cg.eval(&[0.5, 0.8]).unwrap();
let grad = cg.gradient().to_vec();  // [df/dx, df/dy]
```

### `stable_derivatives` for f32 Safety

When computing second-order derivatives that will be evaluated in f32 (e.g., on GPU), use the `stable_derivatives` attribute to automatically route through logarithmic derivative variants:

```rust
#[autodiff(stable_derivatives)]
fn gravity(r2: f64) -> f64 {
    // pow and / are automatically routed to pow_log and div_log
    r2.powf(-1.5) * r2
}
```

This produces identical primal values but avoids catastrophic cancellation in f32 at second order. See [Numerical Precision](numerical_precision.md) for the mathematical details.

### Non-Float Parameters

Parameters typed as `i32` (or any type other than `f64`, `f32`, `Var`) are left untransformed — only float/Var types become generic `T`:

```rust
#[autodiff]
fn power(x: f64, n: i32) -> f64 {
    x.powi(n)
}

// Works with all numeric types — n stays i32
assert_eq!(power(2.0_f64, 3), 8.0);
assert_eq!(power(2.0_f32, 3), 8.0_f32);
```

### Limitations

- **No control flow**: Only transforms expressions, not `if`/`else`, `match`, or loops
- **No comparisons**: `PartialOrd` is not available on `DiffNum`, so comparison operators (`<`, `>`, etc.) cannot be used in generic functions
- **Let bindings**: Variables bound with `let` to float values become `T::from_f64(value)` constants
- **Parameter types**: Only `f64`, `f32`, and `Var` parameter types are replaced with `T`; all other types (e.g., `i32`, `usize`) are left unchanged
- **`powi`/`powi_log`**: The exponent argument must be `i32` (literal or variable), not a float

### f32 Graphs

`AutoDiff` is generic over the float type. For f32 computation graphs:

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::<f32>::new();
let x = ad.var(2.0_f32).unwrap();
let f = ad.square(x);

assert_eq!(ad.eval(f).unwrap(), 4.0_f32);
```

Note: `with_context` / operator overloading is f64-only (because `Var` implements `DiffNum` using a thread-local `AutoDiff<f64>`). For f32 graphs, use `ad.method()` calls or the `expr!` macro.

## Tier 2: `expr!` Declarative Macro

The `expr!` macro provides natural mathematical syntax inline, without requiring the `proc-macros` feature.

### Basic Usage

```rust
use bevy_autodiff::{AutoDiff, expr};

let mut ad = AutoDiff::new();
let x = ad.var(2.0).unwrap();
let y = ad.var(3.0).unwrap();

let f = expr!(ad, x * x + x * y);
assert_eq!(ad.eval(f).unwrap(), 10.0);  // 4 + 6
```

### Supported Syntax

```rust
// Arithmetic with operator precedence
let f = expr!(ad, x * x + 3.0 * x - 1.0);

// Transcendental functions
let f = expr!(ad, sin(x) + cos(x));

// Nested expressions
let f = expr!(ad, exp(x * y) + ln(x));

// Parentheses for grouping
let f = expr!(ad, (1.0 - x) * (1.0 - x));

// Power function
let f = expr!(ad, pow(x, y));
```

### When to Use `expr!`

- One-off formulas that don't need to be reusable functions
- When you don't want the `proc-macros` feature dependency
- Quick prototyping

## Tier 3: Builder API

Direct method calls on `AutoDiff` give full control over graph construction.

### Basic Usage

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::new();
let x = ad.var(2.0).unwrap();

let x_sq = ad.square(x);
let three = ad.constant(3.0);
let three_x = ad.mul(three, x);
let one = ad.constant(1.0);
let sum = ad.add(x_sq, three_x);
let f = ad.add(sum, one);

assert_eq!(ad.eval(f).unwrap(), 11.0);  // f(2) = 4 + 6 + 1
```

### When to Use the Builder API

- **Dynamic graph construction** — building graphs from runtime data (e.g., variable-length sums, programmatic operation sequences)
- **Custom graph manipulation** — when you need to inspect or transform the graph between operations
- **Explicit constant sharing** — reusing `ad.constant()` nodes across multiple expressions

### Symbolic Differentiation (Builder API)

The builder API exposes the symbolic differentiation primitive directly:

```rust
let dfdx = ad.differentiate(f, x).unwrap();
assert_eq!(ad.eval(dfdx).unwrap(), 7.0);  // f'(2) = 2·2 + 3

// Higher-order via successive differentiation
let d2fdx2 = ad.differentiate(dfdx, x).unwrap();
assert_eq!(ad.eval(d2fdx2).unwrap(), 2.0);  // f''(x) = 2
```

## Compilation and Evaluation

All three API tiers produce the same `Var` handles. From there, the compilation and evaluation workflow is identical.

### First-Order Gradient (Reverse-Mode)

Use `compile_primal` + `gradient()` when you need the gradient (all first-order partial derivatives). This is O(1) in the number of inputs — one forward pass + one backward pass regardless of how many inputs you have.

```rust
let mut cg = ad.compile_primal(f, &[x, y]).unwrap();

// Evaluate and get gradient
cg.eval(&[0.5, 0.8]).unwrap();
let val = cg.value();
let grad = cg.gradient();  // [df/dx, df/dy]

// Re-evaluate at a new point without recompiling
cg.eval(&[1.0, 1.0]).unwrap();
let grad = cg.gradient();
```

### Higher-Order Derivatives (Forward-Mode)

Use `compile_order` when you need second-order or higher derivatives. This pre-compiles all partial derivative subgraphs up to the requested order.

```rust
let mut cg = ad.compile_order(f, &[x, y], 2).unwrap();
cg.eval(&[1.0, 2.0]).unwrap();

// Multi-index notation: [order_x, order_y]
let dfdx   = cg.partial(&[1, 0]).unwrap();  // df/dx
let dfdy   = cg.partial(&[0, 1]).unwrap();  // df/dy
let d2fdx2 = cg.partial(&[2, 0]).unwrap();  // d²f/dx²
let d2fdy2 = cg.partial(&[0, 2]).unwrap();  // d²f/dy²
let d2mix  = cg.partial(&[1, 1]).unwrap();  // d²f/dxdy
```

### Choosing a Compilation Method

| Method | Derivatives | Cost per eval | Use case |
|--------|------------|---------------|----------|
| `compile_primal` + `gradient()` | First-order only | O(graph) regardless of inputs | Optimization, sensitivity analysis |
| `compile_order(f, inputs, n)` | Up to order `n` | O(graph × partials) | Hessians, mixed partials |

### Cloning for Multiple Evaluations

`CompiledGraph` implements `Clone`. Build once, clone to evaluate at many independent points:

```rust
let template = ad.compile_primal(f, &[x, y]).unwrap();

let mut cg1 = template.clone();
let mut cg2 = template.clone();

cg1.eval(&[1.0, 2.0]).unwrap();
cg2.eval(&[3.0, 4.0]).unwrap();
// cg1 and cg2 have independent state
```

This is especially useful with Bevy's `par_iter_mut()` for parallel evaluation — see the [Bevy Integration](../README.md#bevy-integration) section.
