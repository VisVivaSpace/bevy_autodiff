# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`bevy_autodiff` is a Rust crate implementing automatic differentiation using Bevy ECS as the computational graph backend. It serves as a baseline AD library exploring what ECS can do for automatic differentiation.

**Core approach:**
- ECS as computation graph: entities are variables, components define operations
- Symbolic graph differentiation: `differentiate(output, wrt)` creates NEW entities representing the derivative graph using the chain rule
- Successive differentiation: for d²f/dxdy, call `differentiate(differentiate(f, x), y)`
- Constant folding during differentiation prevents graph bloat
- Functional/immutable style aligned with DOP principles

**Related crate:** `bevy_entity_ptr` — provides ergonomic entity traversal used by this crate.

## Build Commands

```bash
cargo build                    # Build the library
cargo test                     # Run all tests
cargo test <test_name>         # Run a single test
cargo test -- --nocapture      # Run tests with stdout visible
cargo clippy                   # Run linter
cargo fmt                      # Format code
cargo bench                    # Run benchmarks
```

## Testing

### Unit Tests (default)

```bash
cargo test
```

Runs ~230 tests covering:
- Graph construction and evaluation
- Individual function derivatives (sin, cos, exp, ln, sqrt, etc.)
- Mathematical identities (Pythagorean, exp/ln inverse, etc.)
- Derivative properties (linearity, product rule, mixed partial symmetry)
- Higher-order derivatives via successive differentiation
- CompiledGraph correctness

### Proc-Macro Tests

```bash
cargo test --features proc-macros
```

Tests the `#[autodiff]` attribute macro and `expr!` declarative macro for ergonomic graph construction.

### Oracle Validation: autodiff crate

```bash
cargo test --test autodiff_crate_comparison
```

Compares bevy_autodiff's first-order derivatives against the [autodiff](https://crates.io/crates/autodiff) crate (elrnv/autodiff), an independent forward-mode AD implementation. Covers all supported elementary functions, compositions, and arithmetic combinations.

### Oracle Validation: std::autodiff (Enzyme)

```bash
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme test --features std_autodiff_tests
```

Compares against Rust's experimental `std::autodiff` powered by LLVM/Enzyme. Requires the Enzyme toolchain.

### GPU Tests

```bash
cargo test --features wgpu
```

Tests GPU batch evaluation: NodeOp conversion, dispatch, readback, and GPU-vs-CPU oracle comparison for all 23 operations, compositions, partials, and batch sizes up to 100K. Requires a GPU.

### Test Strategy

| Test Type | What It Validates | Requirements |
|-----------|-------------------|--------------|
| Unit tests | Internal correctness, mathematical identities, derivative properties | None |
| autodiff crate | First derivatives against independent AD | None |
| std::autodiff | First derivatives against LLVM/Enzyme | Enzyme toolchain |
| Proc-macro | Macro expansion and ergonomic API | `proc-macros` feature |
| GPU tests | GPU batch eval against CPU for all ops | `wgpu` feature + GPU |

## Architecture

### Computation Graph as ECS

Variables are entities with components:
- `Variable`, `IsInput`, `IsConstant` — markers
- `Value(f64)` — current numerical value
- `UnaryOp`/`BinaryOp` + `UnaryInput`/`BinaryInputs` — operation definitions
- `Dependencies` — bitmask tracking which inputs affect this variable

### Symbolic Differentiation

`differentiate(output, wrt) -> Var` creates NEW ECS entities representing the derivative graph:

1. Topological sort from output back to inputs
2. For each node, apply the chain rule to create derivative entities
3. Base cases: `d(wrt)/d(wrt) = 1`, `d(other_input)/d(wrt) = 0`, `d(constant)/d(wrt) = 0`
4. Constant folding via smart helpers (`smart_add`, `smart_mul`, etc.) collapses zero/one terms

For higher-order: `differentiate(differentiate(f, x), y)` gives d²f/dxdy.

### CompiledGraph

At compile time, `differentiate()` builds derivative graphs for all requested partials. The entire graph (original + derivatives) is flattened into a `Vec<NodeOp>`. At eval time, a single forward pass computes all values.

### Module Structure

```
src/
├── lib.rs              # Re-exports, crate docs
├── context.rs          # AutoDiff struct, differentiate(), compile(), main API
├── var.rs              # Var handle type
├── compiled.rs         # CompiledGraph, NodeOp, flatten_graph
├── codegen.rs          # WGSL code generation: CompiledGraph::to_wgsl()
├── components/         # ECS components
│   ├── variable.rs     # Variable markers (IsInput, IsConstant, Value, Dependencies)
│   └── operations.rs   # UnaryOp, BinaryOp, UnaryInput, BinaryInputs
├── graph/              # Graph utilities
│   ├── topology.rs     # topological_order, topological_order_multi
│   └── traverse.rs     # Graph traversal utilities
├── gpu/                # GPU batch evaluation (feature = "wgpu")
│   ├── mod.rs          # Module root, re-exports
│   ├── context.rs      # GpuContext: device, queue, pipeline
│   ├── graph.rs        # GpuGraph, GpuResults, dispatch/readback
│   ├── error.rs        # GpuError enum
│   ├── types.rs        # GpuNodeOp, NodeOp→GPU conversion
│   └── shader.wgsl     # WGSL interpreter compute kernel
├── debug.rs            # Graph visualization (DOT format)
├── error.rs            # Error types
├── macros.rs           # expr! macro
├── optimize.rs         # CSE detection, simplification
├── ops.rs              # Operator overloading (Add, Mul, etc.)
└── util.rs             # Math utilities (factorial, binomial)
```

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
bevy_ecs = "0.18"
bevy_entity_ptr = "0.5"
thiserror = "2"
bevy_autodiff-macros = { version = "0.2.0", path = "./bevy_autodiff-macros", optional = true }
wgpu = { version = "28", optional = true }
bytemuck = { version = "1.25", features = ["derive"], optional = true }
pollster = { version = "0.4", optional = true }

[dev-dependencies]
approx = "0.5"    # Floating point comparisons
autodiff = "0.7"  # Oracle validation for derivatives
criterion = "0.5" # Benchmarks
rkf78 = "0.1"     # ODE solver for stm_propagation example

[features]
proc-macros = ["bevy_autodiff-macros"]   # Enable #[autodiff] and expr! macros
std_autodiff_tests = []                  # Enable std::autodiff comparison tests (requires Enzyme)
wgpu = ["dep:wgpu", "dep:bytemuck", "dep:pollster"]  # GPU batch evaluation
```
