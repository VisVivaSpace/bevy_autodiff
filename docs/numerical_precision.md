# Numerical Precision

This document classifies the numerical operations in `bevy_autodiff` by precision tier and justifies the tolerances used in testing. The framework follows the principle that **tolerances must be justified, not guessed**.

## Precision Tiers

### Tier 1: Exact Operations

These produce bit-identical results with no floating-point error:

- `NodeOp::Input(i)` -- copies `inputs[i]` verbatim
- `NodeOp::Constant(v)` -- returns the stored `f64` verbatim
- `BinaryOp::Add`, `BinaryOp::Sub` -- single IEEE 754 operation, exact to 0.5 ULP
- `BinaryOp::Mul` -- single IEEE 754 operation, exact to 0.5 ULP
- `UnaryOp::Neg` -- sign flip, exact

Tests for these operations use `assert_eq!` (exact equality).

### Tier 2: Closed-Form Deterministic

The adjoint helper functions (`unary_adjoint`, `binary_adjoint`) and the forward evaluation functions (`apply_unary_value`, `apply_binary_value`) each evaluate a single closed-form formula using `f64` intrinsics. These are deterministic with bounded relative error:

| Operation | Forward: `f(x)` | Adjoint: `df/dx` | Notes |
|-----------|-----------------|-------------------|-------|
| `sin` | `x.sin()` | `x.cos()` | Single intrinsic each |
| `cos` | `x.cos()` | `-x.sin()` | Single intrinsic each |
| `tan` | `x.tan()` | `1/(cos(x))²` | Adjoint involves 1 intrinsic + 2 muls + 1 div |
| `exp` | `x.exp()` | `z_val` (= `exp(x)`) | Adjoint reuses forward value, exact |
| `ln` | `x.ln()` | `1/x` | Single div |
| `sqrt` | `x.sqrt()` | `0.5/z_val` | Adjoint reuses forward value |
| `sinh` | `x.sinh()` | `x.cosh()` | Single intrinsic |
| `cosh` | `x.cosh()` | `x.sinh()` | Single intrinsic |
| `tanh` | `x.tanh()` | `1 - z²` | Adjoint: 1 mul + 1 sub |
| `asin` | `x.asin()` | `1/sqrt(1-x²)` | Adjoint: 2 muls + 1 sub + 1 sqrt + 1 div |
| `acos` | `x.acos()` | `-1/sqrt(1-x²)` | Same chain as asin |
| `atan` | `x.atan()` | `1/(1+x²)` | 1 mul + 1 add + 1 div |
| `asinh` | `x.asinh()` | `1/sqrt(x²+1)` | 2 ops + sqrt + div |
| `acosh` | `x.acosh()` | `1/sqrt(x²-1)` | 2 ops + sqrt + div; requires `x > 1` |
| `atanh` | `x.atanh()` | `1/(1-x²)` | 1 mul + 1 sub + 1 div |
| `Pow` | `x.powf(y)` | `(y*x^(y-1), z*ln(x))` | Multiple intrinsics |
| `PowLog` | `x.powf(y)` | `(z*y/x, z*ln(x))` | Logarithmic form; see below |
| `Div` | `x/y` | `(1/y, -x/y²)` | 1-2 divs + 1 mul |
| `DivLog` | `x/y` | `(z/x, -z/y)` | Logarithmic form; see below |

For individual adjoint formulas, expected relative error is bounded by a small multiple of machine epsilon (a few ULP). The adjoint unit tests use `epsilon = 1e-12`, which is approximately 4500 ULP -- a generous margin that accounts for the formula chain length while remaining well below any level that would mask a bug. A wrong formula (e.g., using `sin` instead of `cos`) would produce errors of O(1), not O(1e-12).

### Tier 3: Composed Graphs (Reverse-Mode Gradient)

The reverse-mode backward pass chains multiple Tier 2 adjoint evaluations. Error accumulates additively: each adjoint multiplication contributes ~1 ULP of relative error, and these propagate through the `adjoints[src] += adj * local` accumulation.

For a graph of depth `d` with `n` nodes, the gradient error is bounded approximately by `d * epsilon` relative to the partial derivative magnitude. Typical computation graphs have `d < 20`, yielding expected relative errors below `1e-14`.

The cross-validation tests (`test_reverse_vs_forward_*`) compare reverse-mode gradients against forward-mode symbolic partials at multiple evaluation points, using `epsilon = 1e-10`. This tolerance is justified as follows:

- Forward-mode symbolic partials are exact at the level of the compiled graph (they are symbolically derived, not numerically approximated)
- Reverse-mode introduces `O(d * epsilon)` error from chained floating-point adjoint operations
- `1e-10` is ~450,000 ULP -- generous enough to handle deep chains, tight enough to catch formula errors (which produce O(1) disagreement)
- Both modes evaluate the same `f64` intrinsics for forward values, so the comparison isolates adjoint propagation error

### Tier 4: Not Applicable

`bevy_autodiff` does not perform numerical integration or iterative solving. All derivatives are computed symbolically (forward mode) or via exact adjoint formulas (reverse mode). There is no step-size, convergence criterion, or accumulated integration error.

The `stm_propagation` example uses an external integrator (`rkf78`) -- the integration tolerance belongs to that crate, not to `bevy_autodiff`.

## Logarithmic Derivatives and f32 Stability

### The Problem: Catastrophic Cancellation at Second Order

The standard power rule `d(a^b)/da = b · a^(b-1) · da` creates a new node `a^(b-1)`. When this derivative is differentiated again, `a^(b-2)` appears as a separate large value that nearly cancels with `a^(b-1)`-derived terms. In f64, this causes only modest precision loss. In f32 (7 significant digits), the cancellation can produce 100-450% relative errors in second-order derivatives.

This affects any workload that computes Hessians or second-order partials in f32, including GPU batch evaluation of compiled graphs.

### The Solution: Logarithmic Differentiation

The logarithmic alternative rewrites the derivative to reuse the primal value:

| Standard form | Logarithmic form |
|--------------|-----------------|
| `d(a^b)/da = b · a^(b-1) · da` | `d(a^b)/da = a^b · b · (da/a)` |
| `d(a/b)/da = da/b - a·db/b²` | `d(a/b)/da = (a/b) · (da/a - db/b)` |

The logarithmic form works with ratios (`da/a`, `db/b`) rather than creating new large intermediate values. This keeps all intermediate computations within a small factor of the primal values, which is safe for f32's precision.

### Operations

- `pow_log(base, exp)` — same primal as `pow`, logarithmic differentiation rule
- `powi_log(x, n)` — convenience for `pow_log(x, constant(n))`
- `powf_log(x, p)` — convenience for `pow_log(x, constant(p))`
- `div_log(a, b)` — same primal as `div`, logarithmic differentiation rule

These require `base > 0` (for `pow_log`) and `a > 0`, `b > 0` (for `div_log`). When the requirement is violated, the result is NaN — the same behavior as `ln(x)` for `x ≤ 0`.

### When to Use

Use the logarithmic variants when:
- Computing second-order derivatives (Hessians, second partials) that will be evaluated in f32
- Building graphs for GPU batch evaluation where f32 precision matters
- Working with power-law expressions like `r^(-3)` in gravitational or electrostatic problems

The `#[autodiff(stable_derivatives)]` attribute automatically routes all power and division operations to their logarithmic variants.

### Limitation

The logarithmic form eliminates catastrophic cancellation at second order. The intermediate `da/a` sub-expressions use standard division, so at third order and above a milder form of cancellation may appear from differentiating those inner divisions.

## Known Numerical Considerations

### Catastrophic Cancellation in Adjoint Formulas

Several adjoint formulas involve subtraction of nearly-equal quantities near domain boundaries:

- `asin`/`acos` adjoint: `1/sqrt(1 - x²)` -- diverges as `|x| -> 1`
- `atanh` adjoint: `1/(1 - x²)` -- diverges as `|x| -> 1`
- `acosh` adjoint: `1/sqrt(x² - 1)` -- diverges as `x -> 1`
- `tan` adjoint: `1/cos²(x)` -- diverges as `x -> pi/2`

These are inherent to the mathematical derivatives, not numerical artifacts. The adjoint values are correct but large near domain boundaries. Users should avoid evaluating gradients at domain boundaries (e.g., `asin(1.0)`, `atanh(0.999)`).

### Pow Adjoint at x = 0

The `Pow` adjoint `dz/d(rhs) = z * ln(lhs)` produces `-inf` or `NaN` when `lhs = 0` because `ln(0) = -inf`. Similarly, `dz/d(lhs) = rhs * lhs^(rhs-1)` may produce `inf` or `NaN` for `lhs = 0` with `rhs < 1`. This matches the mathematical reality: `x^y` is not differentiable with respect to `y` at `x = 0`.

### IEEE 754 Deviations in Constant Folding

The `smart_mul` and `smart_div` helpers deliberately deviate from IEEE 754:

- `smart_mul`: `0 * x -> 0` even if `x` is NaN or infinity
- `smart_div`: `0 / x -> 0` even if `x` is 0

This is correct for symbolic differentiation: a zero derivative term is structurally zero regardless of the other factor's numerical value. See the doc comments on `smart_mul` and `smart_div` in `context.rs`.

## Test Tolerance Summary

| Test category | Tolerance | Justification |
|---------------|-----------|---------------|
| Adjoint unit tests (single formula) | `1e-12` | Tier 2: few-op closed-form formulas, generous margin over intrinsic precision |
| Manual backward pass tests | `1e-12` | Tier 2-3: short chains of exact operations |
| Cross-validation (reverse vs forward) | `1e-10` | Tier 3: composed graph adjoint chains; forward-mode is the reference |
| Per-operation gradient tests (1D) | `1e-10` | Tier 3: compiled graph forward pass + reverse pass |
| Per-operation gradient tests (2D) | `1e-10`/`1e-12` | Tier 2-3: simple binary ops with minimal chain depth |
| Oracle tests (autodiff crate) | `1e-10` | Cross-implementation: different AD algorithms, same math |
| Forward-mode symbolic tests | `1e-10` | Tier 2-3: symbolic derivatives evaluated at construction-time values |
| Exact arithmetic tests | `assert_eq!` | Tier 1: integer-valued results of exact operations |
| PowLog/DivLog primal match | `assert_eq!` | Tier 1: identical code path as Pow/Div |
| PowLog/DivLog first-order match | `1e-12` rel | Tier 2: closed-form, different formula but same math |
| PowLog/DivLog second-order match | `1e-11` rel | Tier 2: deeper symbolic graph, more folding |
| f32 Hessian (pow_log) | `1e-4` (0.01%) | Problem-dependent: logarithmic form keeps intermediates within ~2.5x of primal |
| Proc-macro derivative tests | `1e-13` | Tier 2: single closed-form derivative evaluation |
