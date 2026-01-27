# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`bevy_autodiff` is a Rust crate implementing lazy, higher-order automatic differentiation using Bevy ECS as the computational graph backend. Full design specification is in `notes/bevy_autodiff_plan.md`.

**Core innovations:**
- ECS as computation graph: entities are variables, components store Taylor coefficients
- Taylor-mode AD: O(n²) complexity for n-th derivative (vs O(exp(n)) for naive nesting)
- Univariate decomposition: avoids multivariate Bell polynomial complexity
- Incremental order growth: compute higher derivatives on demand
- Functional/immutable style aligned with DOP principles

**Related crate:** `bevy_entity_ptr` at `/Users/nstrange/git/vis_viva/clawd/bevy_entity_ptr` — provides ergonomic entity traversal used by this crate.

## Build Commands

```bash
cargo build                    # Build the library
cargo test                     # Run all tests
cargo test <test_name>         # Run a single test
cargo test -- --nocapture      # Run tests with stdout visible
cargo clippy                   # Run linter
cargo fmt                      # Format code
```

## Testing

The test suite validates correctness through multiple complementary approaches:

### Unit Tests (default)

```bash
cargo test
```

Runs ~390 tests covering:
- Taylor coefficient arithmetic (polynomial operations)
- Individual function derivatives (sin, cos, exp, ln, sqrt, etc.)
- Mathematical identities (Pythagorean, exp/ln inverse, etc.)
- Forward and reverse mode agreement
- Higher-order derivatives

### Proc-Macro Tests

```bash
cargo test --features proc-macros
```

Tests the `#[autodiff]` attribute macro and `expr!` declarative macro for ergonomic graph construction.

### Oracle Validation: autodiff crate

```bash
cargo test --test autodiff_crate_comparison
```

Compares bevy_autodiff's first-order derivatives against the [autodiff](https://crates.io/crates/autodiff) crate (elrnv/autodiff), an independent forward-mode AD implementation. This runs without special toolchain requirements and covers all supported functions.

### Oracle Validation: std::autodiff (Enzyme)

```bash
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme test --features std_autodiff_tests
```

Compares against Rust's experimental `std::autodiff` powered by LLVM/Enzyme. Requires building the Enzyme toolchain from source:

```bash
git clone git@github.com:rust-lang/rust
cd rust
./configure --enable-llvm-link-shared --enable-llvm-plugins --enable-llvm-enzyme \
  --release-channel=nightly --enable-llvm-assertions --enable-clang --enable-lld \
  --enable-option-checking --enable-ninja --disable-docs
./x build --stage 1 library
rustup toolchain link enzyme build/host/stage1
```

### Test Strategy

| Test Type | What It Validates | Requirements |
|-----------|-------------------|--------------|
| Unit tests | Internal correctness, mathematical identities | None |
| autodiff crate | First derivatives against independent AD | None |
| std::autodiff | First derivatives against LLVM/Enzyme | Enzyme toolchain |
| Proc-macro | Macro expansion and ergonomic API | `proc-macros` feature |

## Architecture

### Computation Graph as ECS

Variables are entities with components:
- `Variable`, `IsInput`, `IsConstant` — markers
- `Value(f64)` — current numerical value
- `TaylorData` — cached Taylor coefficients per direction
- `UnaryOp`/`BinaryOp` + `UnaryInput`/`BinaryInputs` — operation definitions
- `Dependencies` — bitmask tracking which inputs affect this variable

### Taylor-Mode Strategy

Instead of symbolic differentiation, propagate truncated Taylor polynomials:
1. Parameterize along direction: `p(t) = x + t·d`
2. Compute Taylor series of `f(p(t))`
3. Extract derivatives: `f^(n)(a) = n! · coefficient[n]`

Mixed partials recovered via polarization identity from directional derivatives.

### Planned Module Structure

```
src/
├── lib.rs              # Re-exports, crate docs
├── context.rs          # AutoDiff struct, main API
├── var.rs              # Var handle type
├── components/         # ECS components
│   ├── variable.rs     # Variable markers
│   ├── operations.rs   # UnaryOp, BinaryOp
│   ├── taylor.rs       # TaylorData, Direction, MultiIndex
│   └── adjoint.rs      # AdjointTaylor for reverse mode
├── taylor/             # Forward propagation
│   ├── polynomial.rs   # Truncated polynomial arithmetic
│   ├── propagate.rs    # Graph traversal
│   └── rules/          # Per-operation coefficient rules
├── reverse/            # Reverse accumulation
│   ├── adjoint_rules.rs
│   └── gradient.rs
├── partials/           # Partial derivative extraction
│   ├── directional.rs
│   └── interpolate.rs
└── graph/              # Graph utilities
    ├── topology.rs     # Topological sort
    └── cache.rs        # Invalidation on input change
```

### Key Taylor Coefficient Rules

All rules are O(n²) for order n:
- **Multiplication**: Cauchy product (convolution)
- **Division**: Recurrence solving y·v = u
- **Sin/Cos**: Coupled recurrence (compute together)
- **Exp**: y_k = (1/k) Σ j·u_j·y_{k-j}
- **Ln**: Inverse of exp recurrence

Reference: Griewank & Walther (2008), Tables 13.1-13.2

## Workflow Instructions

1. Write plan to `tasks/todo.md` with checkable items
2. For each step, list code that SHOULD NOT be modified
3. Check in before starting — allow discussion
4. Work through items, marking complete as you go
5. After major phases: add tests, commit, ask for review
6. Add review section to `tasks/todo.md` with summary
7. Find and fix root causes — no temporary fixes
8. Minimal, focused changes only

## Dependencies

```toml
[dependencies]
bevy_ecs = "0.15"
bevy_entity_ptr = { path = "../bevy_entity_ptr" }  # Sibling crate
smallvec = "1.11"
thiserror = "1.0"
bevy_autodiff-macros = { path = "./bevy_autodiff-macros", optional = true }  # Proc-macro crate

[dev-dependencies]
approx = "0.5"   # Floating point comparisons
autodiff = "0.7" # Oracle validation for derivatives

[features]
proc-macros = ["bevy_autodiff-macros"]      # Enable #[autodiff] and expr! macros
std_autodiff_tests = []            # Enable std::autodiff comparison tests (requires Enzyme)
```
