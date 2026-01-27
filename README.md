# bevy_autodiff

Higher-order automatic differentiation using [Bevy ECS](https://bevyengine.org/) as the computational graph backend.

`bevy_autodiff` implements Taylor-mode AD, where variables are ECS entities, operations are components, and Taylor coefficients are propagated through the graph to compute derivatives of any order.

## Key Features

- **ECS as computation graph** — entities are variables, components store operations and Taylor coefficients
- **Taylor-mode AD** — O(n²) complexity for the n-th derivative, vs O(exp(n)) for naive nesting
- **Univariate decomposition** — directional derivatives avoid multivariate Bell polynomial complexity
- **Forward and reverse mode** — forward mode for higher-order derivatives, reverse mode for efficient gradients
- **Incremental order growth** — compute higher derivatives on demand without recomputation
- **Lazy evaluation** — derivatives computed and cached only when requested

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
bevy_autodiff = { git = "https://github.com/VisVivaSpace/bevy_autodiff.git" }
```

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

// Derivatives of any order
assert_eq!(ad.derivative(f, x, 1), 7.0);  // f'(2) = 2·2 + 3
assert_eq!(ad.derivative(f, x, 2), 2.0);  // f''(x) = 2
assert_eq!(ad.derivative(f, x, 3), 0.0);  // f'''(x) = 0
```

## Gradients and Hessians

```rust
use bevy_autodiff::AutoDiff;

let mut ad = AutoDiff::new();
let x = ad.var(1.0);
let y = ad.var(2.0);

// f(x, y) = x² + xy + y²
let x2 = ad.square(x);
let xy = ad.mul(x, y);
let y2 = ad.square(y);
let sum = ad.add(x2, xy);
let f = ad.add(sum, y2);

// Forward-mode gradient
let grad = ad.gradient(f);        // [∂f/∂x, ∂f/∂y] = [4.0, 5.0]

// Reverse-mode gradient (more efficient for many inputs)
let grad_rev = ad.gradient_reverse(f);

// Hessian matrix (second-order partials)
let hessian = ad.hessian(f);
// [[∂²f/∂x², ∂²f/∂x∂y],
//  [∂²f/∂y∂x, ∂²f/∂y²]] = [[2, 1], [1, 2]]
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
bevy_autodiff = { git = "https://github.com/VisVivaSpace/bevy_autodiff.git", features = ["proc-macros"] }
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
let f = rosenbrock(&mut ad, x, y); // Adds `ad` parameter automatically
```

## How It Works

### Taylor-Mode AD

Instead of symbolic differentiation or dual numbers, `bevy_autodiff` propagates truncated Taylor polynomials through the computation graph:

1. **Parameterize** along a direction: `p(t) = x + t·d`
2. **Propagate** Taylor series of `f(p(t))` through each operation using recurrence relations
3. **Extract** derivatives: `f⁽ⁿ⁾(a) = n! · coefficient[n]`

Each operation (exp, sin, mul, etc.) has an O(n²) recurrence relation for computing the n-th Taylor coefficient from lower-order coefficients. Mixed partial derivatives are recovered via the polarization identity from directional derivatives.

### ECS Architecture

The Bevy ECS world stores the computation graph:
- **Entities** represent variables (inputs, constants, intermediate results)
- **Components** store values (`Value`), operations (`UnaryOp`, `BinaryOp`), connectivity (`UnaryInput`, `BinaryInputs`), and cached Taylor coefficients (`TaylorData`)
- **Dependency tracking** via bitmasks identifies which inputs affect each variable

## Examples

Run the included examples:

```bash
cargo run --example basic              # Basic derivatives
cargo run --example gradient           # Forward and reverse gradients
cargo run --example hessian            # Hessian matrix computation
cargo run --example rosenbrock         # Gradient descent optimization
cargo run --example orbital_mechanics  # Gravitational potential derivatives
```

## Testing

The test suite validates correctness through multiple approaches:

```bash
cargo test                           # 351 unit tests + 24 integration tests + doc tests
cargo test --features proc-macros    # Proc-macro tests
cargo test --test autodiff_crate_comparison  # Comparison against autodiff crate
```

Oracle validation against the independent [autodiff](https://crates.io/crates/autodiff) crate ensures first-order derivative correctness. Higher-order derivatives are validated through mathematical identities (Pythagorean, exp/ln inverse, power laws) and forward/reverse mode agreement.

## References

- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed. SIAM. Tables 13.1–13.2 for Taylor coefficient recurrences.

## License

MIT
