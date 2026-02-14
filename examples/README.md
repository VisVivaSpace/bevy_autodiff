# Examples

## Running

```bash
cargo run --example <name>
```

## Overview

### basic

Introductory example: builds a polynomial `f(x) = x² + 3x + 1`, evaluates it, and computes first through third derivatives using symbolic differentiation.

```bash
cargo run --example basic
```

### gradient

Computes the gradient of a multivariate function `f(x,y) = x² + xy + y²` using `ad.gradient()` (forward-mode symbolic partials compiled internally).

```bash
cargo run --example gradient
```

### reverse_gradient

Demonstrates reverse-mode gradient computation using `compile_primal()` + `gradient()`:

1. **Basic gradient**: computes gradient of `f(x,y) = x² + xy + y²`, re-evaluates at multiple points without recompiling
2. **Gradient descent**: minimizes the Rosenbrock function `(1-x)² + 100(y-x²)²` using simple gradient descent with reverse-mode gradients
3. **Three-input function**: `f(x,y,z) = xy + yz + zx` using `eval_gradient()` convenience method

This is the recommended approach for first-order gradients when the number of inputs is large.

```bash
cargo run --example reverse_gradient
```

### hessian

Computes the full Hessian matrix of `f(x,y) = x²y + y³` via successive symbolic differentiation. Demonstrates mixed partial symmetry: `d²f/dxdy = d²f/dydx`.

```bash
cargo run --example hessian
```

### rosenbrock

Rosenbrock function optimization: evaluates `f(x,y) = (1-x)² + 100(y-x²)²` and its gradient at multiple points. Demonstrates both `ad.gradient()` (forward-mode) and `CompiledGraph` with `compile_order` for fast repeated evaluation.

```bash
cargo run --example rosenbrock
```

### orbital_mechanics

Computes the gravitational potential `V = -mu/r` and its gradient (acceleration vector) for a point mass. Demonstrates `bevy_autodiff` in an astrodynamics context where analytical derivatives of the force model are needed.

```bash
cargo run --example orbital_mechanics
```

### stm_propagation

Full State Transition Matrix (STM) propagation for a two-body orbit. This is the most complete aerospace example:

1. Builds the gravity gradient Jacobian symbolically with `bevy_autodiff`
2. Compiles the derivatives once before integration
3. Evaluates them at each integration step inside an `rkf78` ODE solver
4. Propagates the 6x6 STM alongside a circular LEO orbit for one period
5. Verifies: Jacobian against analytical formula, det(STM) = 1 (Liouville), and STM perturbation prediction against actual re-propagation

Requires the `rkf78` dev-dependency.

```bash
cargo run --example stm_propagation
```

### gpu_batch

GPU batch evaluation: compiles `f(x, y) = sin(x·y) + exp(x)` with first-order partials, then evaluates at 1 million input points in parallel on the GPU via wgpu. Demonstrates the full GPU workflow: `GpuContext::new()`, `prepare()`, `eval_batch()`, and reading back values and partial derivatives.

Requires the `wgpu` feature.

```bash
cargo run --example gpu_batch --features wgpu
```
