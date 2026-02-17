# Architecture

## Overview

`bevy_autodiff` uses Bevy ECS as the computational graph backend for automatic differentiation. The graph is built imperatively by calling methods on `AutoDiff`, which spawns entities with components that define operations and connectivity. Derivatives are computed by symbolic graph differentiation (creating new entities via the chain rule), or by reverse-mode adjoint propagation over a compiled flat array.

## Graph Representation

### ECS Layer

Every node in the computation graph is a Bevy ECS entity with a subset of these components:

| Component | Purpose |
|-----------|---------|
| `Variable` | Marker: this entity is part of the computation graph |
| `IsInput` | Marker: this is an input variable (leaf, user-controlled) |
| `IsConstant` | Marker: this is a fixed constant (leaf, zero derivative) |
| `Value<F>` | Current numerical value (generic over float type) |
| `UnaryOpMarker(UnaryOp)` | Which unary operation this node represents |
| `BinaryOpMarker(BinaryOp)` | Which binary operation this node represents (`Add`, `Sub`, `Mul`, `Div`, `Pow`, `DivLog`, `PowLog`) |
| `UnaryInput(EntityHandle)` | The single input entity for a unary op |
| `BinaryInputs { left, right }` | The two input entities for a binary op |
| `Dependencies(u64)` | Bitmask tracking which input variables affect this node |

The `Dependencies` bitmask supports up to 64 input variables. Each input variable is assigned a unique bit at creation time.

### Compiled Layer (CompiledGraph)

For repeated evaluation, the ECS graph is flattened into a `Vec<NodeOp>` -- a topologically sorted array where each node is one of:

```rust
enum NodeOp<F> {
    Input(usize),                         // Read from inputs[index]
    Constant(F),                          // Fixed value
    Unary { op: UnaryOp, src: usize },    // op(nodes[src])
    Binary { op: BinaryOp, lhs: usize, rhs: usize },  // op(nodes[lhs], nodes[rhs])
}
```

All references are by array index, not entity ID. A single forward pass over this array evaluates the entire graph. The values are cached for subsequent reverse-mode gradient computation.

## Differentiation Approaches

### Forward-Mode Symbolic (higher-order capable)

`differentiate(output, wrt)` creates new ECS entities representing the derivative graph:

1. Topological sort from `output` back to inputs
2. For each node, apply the chain rule to produce a derivative entity
3. Base cases: `d(wrt)/d(wrt) = 1`, `d(other)/d(wrt) = 0`
4. Constant folding (`smart_add`, `smart_mul`, etc.) eliminates zero/one terms

For d²f/dxdy: call `differentiate(differentiate(f, x), y)`. Each call adds entities to the same ECS world. The resulting derivative entities can be compiled alongside the primal into a single `CompiledGraph` via `compile()` or `compile_order()`.

**Cost**: O(graph_size) per partial derivative. Scales linearly with the number of requested partials.

### Reverse-Mode Adjoint (first-order, all inputs at once)

`CompiledGraph::gradient()` computes the full gradient via adjoint propagation:

1. Forward pass: `eval()` caches all node values in `values[]`
2. Backward pass: walk `(0..=output_node).rev()`, propagating adjoints via local partial derivatives (`unary_adjoint`, `binary_adjoint`)
3. Seed: `adjoints[output_node] = 1.0`
4. Gather: read `adjoints[input_node]` for each input

**Cost**: O(graph_size) total for all partial derivatives, regardless of input count.

**When to use which**:
- Reverse-mode (`compile_primal` + `gradient`): first-order gradients, especially with many inputs
- Forward-mode (`compile_order`): second-order or higher derivatives, mixed partials, Hessians

## Module Structure

```
src/
  lib.rs              # Crate root, re-exports
  context.rs          # AutoDiff<F> struct: graph building, differentiate(), compile()
  var.rs              # Var handle type (wraps Entity)
  compiled.rs         # CompiledGraph<F>, NodeOp<F>, flatten_graph, adjoint helpers
  codegen.rs          # WGSL code generation from CompiledGraph
  diff_num.rs         # DiffNum and Float traits
  error.rs            # AutoDiffError enum
  components/
    variable.rs       # Variable, IsInput, IsConstant, Value<F>, Dependencies
    operations.rs     # UnaryOp, BinaryOp, UnaryOpMarker, BinaryOpMarker, inputs
  graph/
    topology.rs       # topological_order, topological_order_multi
    traverse.rs       # Graph traversal utilities
  gpu/                # GPU batch evaluation (feature = "wgpu")
    context.rs        # GpuContext: device, queue, pipeline
    graph.rs          # GpuGraph, GpuResults, dispatch/readback
    error.rs          # GpuError enum
    types.rs          # GpuNodeOp, NodeOp→GPU conversion
    shader.wgsl       # WGSL interpreter compute kernel
  debug.rs            # DOT format visualization, validation
  macros.rs           # expr! macro
  ops.rs              # Operator overloading (Add, Mul, etc. for Var)
```

## Compilation Pipeline

```
AutoDiff (ECS world)
  |
  |-- differentiate(f, x)     creates derivative entities in the same world
  |-- differentiate(df, y)    creates second-derivative entities
  |
  v
compile(output, inputs, partials)
  |
  |-- topological_order_multi()   finds all reachable entities
  |-- flatten_graph()             maps entities to NodeOp indices
  |
  v
CompiledGraph (Vec<NodeOp>)
  |
  |-- eval(inputs)       forward pass: fills values[]
  |-- value()            read function value
  |-- partial(&[1,0])    read pre-compiled symbolic derivative
  |-- gradient()         reverse pass: fills adjoints[], returns gradient
```

## Constant Folding

During symbolic differentiation, the `smart_*` helpers prevent graph bloat:

- `smart_add(a, b)`: if `a` is constant 0, return `b` (and vice versa)
- `smart_mul(a, b)`: if either is constant 0, return 0; if either is constant 1, return the other; if either is constant -1, return `neg` of the other
- `smart_neg(a)`: if `a` is constant 0, return `a`
- `smart_div(a, b)`: if numerator is constant 0, return 0; if denominator is constant 1, return numerator; if `a` and `b` are the same entity, return 1
- `smart_sub(a, b)`: if `b` is constant 0, return `a`; if `a` is constant 0, negate `b`; if `a` and `b` are the same entity, return 0

These deliberately deviate from IEEE 754 (where `0 * NaN = NaN`) because in symbolic differentiation, a zero derivative term is structurally zero regardless of the other factor's value. This prevents NaN poisoning the derivative graph when subexpressions hit domain boundaries.

## Bevy Integration

`CompiledGraph` derives `Clone`, `Component`, and `Resource`, enabling direct use in a Bevy application's ECS. The integration follows a "shallow" pattern:

**Graph construction** happens in `AutoDiff`'s private `World`. This isolates the computation graph from the application's ECS — no contention, no component pollution, no waiting on unrelated systems.

**Compiled evaluation** crosses into the app's ECS via `CompiledGraph` as a `Component`. Each entity gets its own clone with independent evaluation state (`values[]`, `adjoints[]`, `gradient_buf[]`).

**Parallel evaluation** uses Bevy's `ComputeTaskPool` via `par_iter_mut()`. Since `eval(&mut self)` takes exclusive mutable access, the scheduler can safely distribute evaluation across threads:

```
AutoDiff (private World)           App World (Bevy scheduler)
  |                                  |
  |-- build graph                    |-- spawn entities with CompiledGraph
  |-- compile() → CompiledGraph      |-- par_iter_mut() → parallel eval()
  |                                  |-- gradient() per entity
```

**GPU types**: `GpuContext` derives `Resource` (singleton device/queue/pipeline). `GpuGraph` derives `Component` and `Resource` (per-graph GPU buffers). `GpuResults` has no derives — it's an ephemeral return value.
