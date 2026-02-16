//! Compiled computation graph for fast repeated evaluation.
//!
//! `CompiledGraph` flattens the ECS graph (and its derivative subgraphs)
//! into a `Vec<NodeOp>` that can be re-evaluated at new input values
//! without touching the ECS world.
//!
//! At compile time, `differentiate()` builds derivative graphs for all
//! requested partials. The entire graph (original + derivatives) is then
//! flattened into one forward-pass array.

use std::collections::HashMap;

use bevy_ecs::entity::Entity;

use crate::components::{
    BinaryInputs, BinaryOp, BinaryOpMarker, IsConstant, IsInput, UnaryInput, UnaryOp,
    UnaryOpMarker, Value,
};

/// A node in the flattened computation graph.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum NodeOp {
    /// Read from `inputs[index]`.
    Input(usize),
    /// Fixed constant value.
    Constant(f64),
    /// Unary operation on node at index `src`.
    Unary { op: UnaryOp, src: usize },
    /// Binary operation on nodes at indices `lhs`, `rhs`.
    Binary {
        op: BinaryOp,
        lhs: usize,
        rhs: usize,
    },
}

/// A compiled computation graph for fast repeated evaluation.
///
/// Created by [`AutoDiff::compile`](crate::AutoDiff::compile),
/// [`AutoDiff::compile_order`](crate::AutoDiff::compile_order), or
/// [`AutoDiff::compile_primal`](crate::AutoDiff::compile_primal).
/// Stores a flattened, topologically sorted
/// node array that can be re-evaluated at new input values without
/// touching the ECS world.
///
/// # Evaluation modes
///
/// - **Forward-mode symbolic partials**: When compiled with `compile()` or
///   `compile_order()`, derivative subgraphs are pre-built. Use [`eval`](Self::eval)
///   + [`partial`](Self::partial) to read pre-computed derivatives.
///
/// - **Reverse-mode gradient**: When compiled with `compile_primal()` (or any
///   compile method), use [`eval`](Self::eval) + [`gradient`](Self::gradient) to
///   compute the full gradient via a single backward pass. Cost is independent
///   of the number of inputs.
///
/// # Bevy integration
///
/// `CompiledGraph` derives [`Clone`], [`bevy_ecs::component::Component`], and
/// [`bevy_ecs::resource::Resource`]. Attach cloned graphs to entities and use
/// `Query::par_iter_mut()` for parallel evaluation across Bevy's `ComputeTaskPool`.
#[derive(Clone, bevy_ecs::component::Component, bevy_ecs::resource::Resource)]
pub struct CompiledGraph {
    nodes: Vec<NodeOp>,
    num_inputs: usize,
    output_index: usize,
    partial_outputs: Vec<(Vec<usize>, usize)>,
    partial_lookup: HashMap<Vec<usize>, usize>,
    values: Vec<f64>,
    /// Adjoint buffer for reverse-mode gradient computation.
    adjoints: Vec<f64>,
    /// Maps input position i to the node index of `NodeOp::Input(i)`.
    input_node_indices: Vec<usize>,
    /// Reusable output buffer for gradient results (length = num_inputs).
    gradient_buf: Vec<f64>,
    /// Whether eval() has been called at least once.
    has_evaluated: bool,
}

impl std::fmt::Debug for CompiledGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("nodes", &self.nodes.len())
            .field("num_inputs", &self.num_inputs)
            .field("partials", &self.partial_outputs.len())
            .finish()
    }
}

impl CompiledGraph {
    /// Creates a new CompiledGraph from pre-built components.
    pub(crate) fn new(
        nodes: Vec<NodeOp>,
        num_inputs: usize,
        output_index: usize,
        partial_outputs: Vec<(Vec<usize>, usize)>,
    ) -> Self {
        let num_nodes = nodes.len();
        let partial_lookup: HashMap<Vec<usize>, usize> = partial_outputs
            .iter()
            .map(|(mi, idx)| (mi.clone(), *idx))
            .collect();

        // Build input_node_indices: for each input position 0..num_inputs,
        // find the node index that contains NodeOp::Input(pos).
        let mut input_node_indices = vec![0usize; num_inputs];
        for (i, node) in nodes.iter().enumerate() {
            if let NodeOp::Input(pos) = node {
                input_node_indices[*pos] = i;
            }
        }

        Self {
            nodes,
            num_inputs,
            output_index,
            partial_outputs,
            partial_lookup,
            values: vec![0.0; num_nodes],
            adjoints: vec![0.0; num_nodes],
            input_node_indices,
            gradient_buf: vec![0.0; num_inputs],
            has_evaluated: false,
        }
    }

    /// Evaluates the compiled graph at the given input values.
    ///
    /// After calling this, use `value()` and `partial()` to read results.
    ///
    /// # Errors
    ///
    /// Returns [`InputCountMismatch`](crate::error::AutoDiffError::InputCountMismatch) if the number of inputs
    /// does not match the compiled graph's expected input count.
    pub fn eval(&mut self, inputs: &[f64]) -> Result<(), crate::error::AutoDiffError> {
        if inputs.len() != self.num_inputs {
            return Err(crate::error::AutoDiffError::InputCountMismatch {
                expected: self.num_inputs,
                got: inputs.len(),
            });
        }

        for i in 0..self.nodes.len() {
            self.values[i] = match self.nodes[i] {
                NodeOp::Input(idx) => inputs[idx],
                NodeOp::Constant(v) => v,
                NodeOp::Unary { op, src } => apply_unary_value(op, self.values[src]),
                NodeOp::Binary { op, lhs, rhs } => {
                    apply_binary_value(op, self.values[lhs], self.values[rhs])
                }
            };
        }
        self.has_evaluated = true;
        Ok(())
    }

    /// Returns the function value from the most recent `eval()`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `eval()` has been called at least once.
    #[inline]
    pub fn value(&self) -> f64 {
        debug_assert!(
            self.has_evaluated,
            "CompiledGraph::value() called before eval()"
        );
        self.values[self.output_index]
    }

    /// Returns a partial derivative from the most recent `eval()`.
    ///
    /// The `multi_index` must match one of the partials passed to `compile()`.
    ///
    /// # Errors
    ///
    /// Returns [`PartialNotCompiled`](crate::error::AutoDiffError::PartialNotCompiled) if the requested partial
    /// was not included when the graph was compiled.
    pub fn partial(&self, multi_index: &[usize]) -> Result<f64, crate::error::AutoDiffError> {
        debug_assert!(
            self.has_evaluated,
            "CompiledGraph::partial() called before eval()"
        );
        if let Some(&idx) = self.partial_lookup.get(multi_index) {
            return Ok(self.values[idx]);
        }
        Err(crate::error::AutoDiffError::PartialNotCompiled {
            requested: multi_index.to_vec(),
            available: self
                .partial_outputs
                .iter()
                .map(|(mi, _)| mi.clone())
                .collect(),
        })
    }

    /// Returns the number of nodes in the compiled graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of inputs this graph expects.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Returns a reference to the node array.
    pub(crate) fn nodes(&self) -> &[NodeOp] {
        &self.nodes
    }

    /// Returns the index of the primary output node.
    pub(crate) fn output_index(&self) -> usize {
        self.output_index
    }

    /// Returns the compiled partial outputs as (multi_index, node_index) pairs.
    pub(crate) fn partial_outputs(&self) -> &[(Vec<usize>, usize)] {
        &self.partial_outputs
    }

    /// Returns all available partial multi-indices.
    pub fn available_partials(&self) -> Vec<Vec<usize>> {
        self.partial_outputs
            .iter()
            .map(|(mi, _)| mi.clone())
            .collect()
    }

    // =========================================================================
    // Reverse-mode gradient computation
    // =========================================================================

    /// Computes the gradient of the primary output with respect to all inputs
    /// via a reverse-mode backward pass.
    ///
    /// Must call `eval()` first so that forward values are populated.
    /// Returns a slice of length `num_inputs` where element `i` is
    /// `∂output/∂input_i`.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `eval()` has been called at least once.
    pub fn gradient(&mut self) -> &[f64] {
        debug_assert!(
            self.has_evaluated,
            "CompiledGraph::gradient() called before eval()"
        );
        self.gradient_of(self.output_index)
    }

    /// Computes the gradient of an arbitrary node with respect to all inputs
    /// via a reverse-mode backward pass.
    ///
    /// Must call `eval()` first so that forward values are populated.
    /// `output_node` is the index into the node array whose gradient is desired.
    /// Returns a slice of length `num_inputs`.
    pub fn gradient_of(&mut self, output_node: usize) -> &[f64] {
        // 1. Zero the adjoint buffer
        for a in self.adjoints.iter_mut() {
            *a = 0.0;
        }

        // 2. Seed the output node
        self.adjoints[output_node] = 1.0;

        // 3. Reverse sweep: walk from output_node down to node 0
        for i in (0..=output_node).rev() {
            let adj = self.adjoints[i];
            // Bitwise zero check is sound here: adjoints are initialized to 0.0
            // and only modified by seeding (= 1.0) or accumulation (+= adj * local).
            // A node with adj == 0.0 contributes nothing downstream, so we skip it.
            if adj == 0.0 {
                continue;
            }

            match self.nodes[i] {
                NodeOp::Input(_) | NodeOp::Constant(_) => {
                    // Leaf nodes: nothing to propagate
                }
                NodeOp::Unary { op, src } => {
                    let local = unary_adjoint(op, self.values[src], self.values[i]);
                    self.adjoints[src] += adj * local;
                }
                NodeOp::Binary { op, lhs, rhs } => {
                    let (dl, dr) =
                        binary_adjoint(op, self.values[lhs], self.values[rhs], self.values[i]);
                    self.adjoints[lhs] += adj * dl;
                    self.adjoints[rhs] += adj * dr;
                }
            }
        }

        // 4. Gather input adjoints into gradient buffer
        for (i, &node_idx) in self.input_node_indices.iter().enumerate() {
            self.gradient_buf[i] = self.adjoints[node_idx];
        }

        &self.gradient_buf
    }

    /// Evaluates the graph at the given inputs and computes the gradient
    /// of the primary output in one call.
    ///
    /// Equivalent to calling `eval(inputs)` followed by `gradient()`.
    /// Returns a slice of length `num_inputs`.
    pub fn eval_gradient(&mut self, inputs: &[f64]) -> Result<&[f64], crate::error::AutoDiffError> {
        self.eval(inputs)?;
        Ok(self.gradient())
    }
}

// =============================================================================
// Compilation from ECS graph
// =============================================================================

/// Flattens a set of ECS entities (topologically ordered) into a `Vec<NodeOp>`.
///
/// Returns `(nodes, entity_to_index)`.
pub(crate) fn flatten_graph(
    world: &bevy_ecs::world::World,
    order: &[Entity],
    input_to_pos: &HashMap<Entity, usize>,
) -> (Vec<NodeOp>, HashMap<Entity, usize>) {
    let entity_to_index: HashMap<Entity, usize> =
        order.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    let mut nodes = Vec::with_capacity(order.len());
    for &entity in order {
        let entity_ref = world.entity(entity);

        if entity_ref.contains::<IsInput>() {
            if let Some(&pos) = input_to_pos.get(&entity) {
                nodes.push(NodeOp::Input(pos));
            } else {
                // Input not in our compile list — freeze at current value
                let val = entity_ref
                    .get::<Value>()
                    .expect("internal: IsInput entity must have Value")
                    .get();
                nodes.push(NodeOp::Constant(val));
            }
        } else if entity_ref.contains::<IsConstant>() {
            let val = entity_ref
                .get::<Value>()
                .expect("internal: IsConstant entity must have Value")
                .get();
            nodes.push(NodeOp::Constant(val));
        } else if let Some(&UnaryOpMarker(op)) = entity_ref.get::<UnaryOpMarker>() {
            let src_entity = entity_ref
                .get::<UnaryInput>()
                .expect("internal: UnaryOpMarker entity must have UnaryInput")
                .get()
                .entity();
            let src = entity_to_index[&src_entity];
            nodes.push(NodeOp::Unary { op, src });
        } else if let Some(&BinaryOpMarker(op)) = entity_ref.get::<BinaryOpMarker>() {
            let binary = entity_ref
                .get::<BinaryInputs>()
                .expect("internal: BinaryOpMarker entity must have BinaryInputs");
            let lhs = entity_to_index[&binary.left.entity()];
            let rhs = entity_to_index[&binary.right.entity()];
            nodes.push(NodeOp::Binary { op, lhs, rhs });
        } else {
            // Fallback: use current value as constant
            let val = entity_ref.get::<Value>().map(|v| v.get()).unwrap_or(0.0);
            nodes.push(NodeOp::Constant(val));
        }
    }

    (nodes, entity_to_index)
}

// =============================================================================
// Multi-index generation
// =============================================================================

/// Generates all multi-indices of dimension `n` with total order 1..=max_order.
pub(crate) fn generate_multi_indices(n: usize, max_order: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = vec![0usize; n];
    generate_helper(n, max_order, 0, &mut current, &mut result);
    result
}

fn generate_helper(
    n: usize,
    max_total: usize,
    pos: usize,
    current: &mut [usize],
    result: &mut Vec<Vec<usize>>,
) {
    if pos == n {
        let total: usize = current.iter().sum();
        if total > 0 && total <= max_total {
            result.push(current.to_vec());
        }
        return;
    }
    let used: usize = current[..pos].iter().sum();
    for k in 0..=(max_total - used) {
        current[pos] = k;
        generate_helper(n, max_total, pos + 1, current, result);
    }
}

// =============================================================================
// Value-level helpers (compute f(x) for each operation, no derivatives)
// =============================================================================

/// Apply a unary operation to a value.
pub(crate) fn apply_unary_value(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Asin => x.asin(),
        UnaryOp::Acos => x.acos(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Asinh => x.asinh(),
        UnaryOp::Acosh => x.acosh(),
        UnaryOp::Atanh => x.atanh(),
    }
}

/// Apply a binary operation to two values.
pub(crate) fn apply_binary_value(op: BinaryOp, x: f64, y: f64) -> f64 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div | BinaryOp::DivLog => x / y,
        BinaryOp::Pow | BinaryOp::PowLog => x.powf(y),
    }
}

// =============================================================================
// Adjoint helpers (local partial derivatives for reverse mode)
// =============================================================================

/// Local partial derivative of z = op(src) with respect to src.
///
/// Given the forward values `src_val` and `z_val = op(src_val)`,
/// returns dz/d(src).
pub(crate) fn unary_adjoint(op: UnaryOp, src_val: f64, z_val: f64) -> f64 {
    match op {
        UnaryOp::Neg => -1.0,
        UnaryOp::Sin => src_val.cos(),
        UnaryOp::Cos => -src_val.sin(),
        UnaryOp::Tan => {
            let c = src_val.cos();
            1.0 / (c * c)
        }
        UnaryOp::Exp => z_val,
        UnaryOp::Ln => 1.0 / src_val,
        UnaryOp::Sqrt => 0.5 / z_val,
        UnaryOp::Sinh => src_val.cosh(),
        UnaryOp::Cosh => src_val.sinh(),
        UnaryOp::Tanh => 1.0 - z_val * z_val,
        UnaryOp::Asin => 1.0 / (1.0 - src_val * src_val).sqrt(),
        UnaryOp::Acos => -1.0 / (1.0 - src_val * src_val).sqrt(),
        UnaryOp::Atan => 1.0 / (1.0 + src_val * src_val),
        UnaryOp::Asinh => 1.0 / (src_val * src_val + 1.0).sqrt(),
        UnaryOp::Acosh => 1.0 / (src_val * src_val - 1.0).sqrt(),
        UnaryOp::Atanh => 1.0 / (1.0 - src_val * src_val),
    }
}

/// Local partial derivatives of z = op(lhs, rhs) with respect to (lhs, rhs).
///
/// Given the forward values `lhs_val`, `rhs_val`, and `z_val = op(lhs_val, rhs_val)`,
/// returns (dz/d(lhs), dz/d(rhs)).
pub(crate) fn binary_adjoint(op: BinaryOp, lhs_val: f64, rhs_val: f64, z_val: f64) -> (f64, f64) {
    match op {
        BinaryOp::Add => (1.0, 1.0),
        BinaryOp::Sub => (1.0, -1.0),
        BinaryOp::Mul => (rhs_val, lhs_val),
        BinaryOp::Div => (1.0 / rhs_val, -lhs_val / (rhs_val * rhs_val)),
        BinaryOp::Pow => {
            // dz/d(lhs) = rhs * lhs^(rhs-1)
            let dlhs = rhs_val * lhs_val.powf(rhs_val - 1.0);
            // dz/d(rhs) = z * ln(lhs)
            let drhs = z_val * lhs_val.ln();
            (dlhs, drhs)
        }
        BinaryOp::PowLog => {
            // Logarithmic form: dz/d(lhs) = z * rhs / lhs
            let dlhs = z_val * rhs_val / lhs_val;
            // dz/d(rhs) = z * ln(lhs)
            let drhs = z_val * lhs_val.ln();
            (dlhs, drhs)
        }
        BinaryOp::DivLog => {
            // Logarithmic form: same as standard Div for reverse mode
            (1.0 / rhs_val, -lhs_val / (rhs_val * rhs_val))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_multi_indices_1d() {
        let indices = generate_multi_indices(1, 3);
        assert_eq!(indices, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_generate_multi_indices_2d_order1() {
        let indices = generate_multi_indices(2, 1);
        assert!(indices.contains(&vec![1, 0]));
        assert!(indices.contains(&vec![0, 1]));
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_generate_multi_indices_2d_order2() {
        let indices = generate_multi_indices(2, 2);
        // Order 1: [1,0], [0,1]
        // Order 2: [2,0], [1,1], [0,2]
        assert_eq!(indices.len(), 5);
        assert!(indices.contains(&vec![1, 0]));
        assert!(indices.contains(&vec![0, 1]));
        assert!(indices.contains(&vec![2, 0]));
        assert!(indices.contains(&vec![1, 1]));
        assert!(indices.contains(&vec![0, 2]));
    }

    #[test]
    fn test_compiled_graph_eval_simple() {
        // Manual construction: f(x) = 2*x + 1
        let nodes = vec![
            NodeOp::Input(0),      // node 0: x
            NodeOp::Constant(2.0), // node 1: 2
            NodeOp::Binary {
                // node 2: 2*x
                op: BinaryOp::Mul,
                lhs: 1,
                rhs: 0,
            },
            NodeOp::Constant(1.0), // node 3: 1
            NodeOp::Binary {
                // node 4: 2*x + 1
                op: BinaryOp::Add,
                lhs: 2,
                rhs: 3,
            },
        ];

        let mut cg = CompiledGraph::new(nodes, 1, 4, vec![]);
        cg.eval(&[3.0]).unwrap();
        assert_eq!(cg.value(), 7.0); // 2*3 + 1
        cg.eval(&[5.0]).unwrap();
        assert_eq!(cg.value(), 11.0); // 2*5 + 1
    }

    // =========================================================================
    // Adjoint helper unit tests
    // =========================================================================

    use approx::assert_relative_eq;

    #[test]
    fn test_unary_adjoint_neg() {
        assert_eq!(unary_adjoint(UnaryOp::Neg, 3.0, -3.0), -1.0);
    }

    #[test]
    fn test_unary_adjoint_sin() {
        let x: f64 = 0.7;
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Sin, x, x.sin()),
            x.cos(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_cos() {
        let x: f64 = 0.7;
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Cos, x, x.cos()),
            -x.sin(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_tan() {
        let x: f64 = 0.7;
        let expected = 1.0 / (x.cos() * x.cos());
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Tan, x, x.tan()),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_exp() {
        let x: f64 = 1.5;
        let z = x.exp();
        assert_relative_eq!(unary_adjoint(UnaryOp::Exp, x, z), z, epsilon = 1e-12);
    }

    #[test]
    fn test_unary_adjoint_ln() {
        let x: f64 = 2.0;
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Ln, x, x.ln()),
            1.0 / x,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_sqrt() {
        let x: f64 = 4.0;
        let z = x.sqrt();
        assert_relative_eq!(unary_adjoint(UnaryOp::Sqrt, x, z), 0.5 / z, epsilon = 1e-12);
    }

    #[test]
    fn test_unary_adjoint_sinh() {
        let x: f64 = 1.0;
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Sinh, x, x.sinh()),
            x.cosh(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_cosh() {
        let x: f64 = 1.0;
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Cosh, x, x.cosh()),
            x.sinh(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_tanh() {
        let x: f64 = 0.5;
        let z = x.tanh();
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Tanh, x, z),
            1.0 - z * z,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_asin() {
        let x: f64 = 0.5;
        let expected = 1.0 / (1.0 - x * x).sqrt();
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Asin, x, x.asin()),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_acos() {
        let x: f64 = 0.5;
        let expected = -1.0 / (1.0 - x * x).sqrt();
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Acos, x, x.acos()),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_atan() {
        let x: f64 = 1.0;
        let expected = 1.0 / (1.0 + x * x);
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Atan, x, x.atan()),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_asinh() {
        let x: f64 = 1.0;
        let expected = 1.0 / (x * x + 1.0).sqrt();
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Asinh, x, x.asinh()),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_acosh() {
        let x: f64 = 2.0;
        let expected = 1.0 / (x * x - 1.0).sqrt();
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Acosh, x, x.acosh()),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_unary_adjoint_atanh() {
        let x: f64 = 0.5;
        let expected = 1.0 / (1.0 - x * x);
        assert_relative_eq!(
            unary_adjoint(UnaryOp::Atanh, x, x.atanh()),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_binary_adjoint_add() {
        assert_eq!(binary_adjoint(BinaryOp::Add, 2.0, 3.0, 5.0), (1.0, 1.0));
    }

    #[test]
    fn test_binary_adjoint_sub() {
        assert_eq!(binary_adjoint(BinaryOp::Sub, 5.0, 3.0, 2.0), (1.0, -1.0));
    }

    #[test]
    fn test_binary_adjoint_mul() {
        assert_eq!(binary_adjoint(BinaryOp::Mul, 2.0, 3.0, 6.0), (3.0, 2.0));
    }

    #[test]
    fn test_binary_adjoint_div() {
        let (dl, dr) = binary_adjoint(BinaryOp::Div, 6.0, 3.0, 2.0);
        assert_relative_eq!(dl, 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(dr, -6.0 / 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_binary_adjoint_pow() {
        // z = 2^3 = 8
        let (dl, dr) = binary_adjoint(BinaryOp::Pow, 2.0, 3.0, 8.0);
        // dz/dlhs = 3 * 2^2 = 12
        assert_relative_eq!(dl, 12.0, epsilon = 1e-12);
        // dz/drhs = 8 * ln(2)
        assert_relative_eq!(dr, 8.0 * 2.0_f64.ln(), epsilon = 1e-12);
    }

    // =========================================================================
    // Reverse-mode backward pass tests (manual graphs)
    // =========================================================================

    #[test]
    fn test_gradient_linear() {
        // f(x) = 2*x + 1, df/dx = 2
        let nodes = vec![
            NodeOp::Input(0),      // node 0: x
            NodeOp::Constant(2.0), // node 1: 2
            NodeOp::Binary {
                op: BinaryOp::Mul,
                lhs: 1,
                rhs: 0,
            }, // node 2: 2*x
            NodeOp::Constant(1.0), // node 3: 1
            NodeOp::Binary {
                op: BinaryOp::Add,
                lhs: 2,
                rhs: 3,
            }, // node 4: 2*x + 1
        ];

        let mut cg = CompiledGraph::new(nodes, 1, 4, vec![]);
        cg.eval(&[3.0]).unwrap();
        let grad = cg.gradient();
        assert_relative_eq!(grad[0], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gradient_two_inputs() {
        // f(x, y) = x * y, df/dx = y, df/dy = x
        let nodes = vec![
            NodeOp::Input(0), // node 0: x
            NodeOp::Input(1), // node 1: y
            NodeOp::Binary {
                op: BinaryOp::Mul,
                lhs: 0,
                rhs: 1,
            }, // node 2: x*y
        ];

        let mut cg = CompiledGraph::new(nodes, 2, 2, vec![]);
        cg.eval(&[3.0, 5.0]).unwrap();
        let grad = cg.gradient();
        assert_relative_eq!(grad[0], 5.0, epsilon = 1e-12); // df/dx = y
        assert_relative_eq!(grad[1], 3.0, epsilon = 1e-12); // df/dy = x
    }

    #[test]
    fn test_gradient_shared_subexpr() {
        // f(x) = x * x, df/dx = 2x
        // Both lhs and rhs of mul point to the same input node
        let nodes = vec![
            NodeOp::Input(0), // node 0: x
            NodeOp::Binary {
                op: BinaryOp::Mul,
                lhs: 0,
                rhs: 0,
            }, // node 1: x*x
        ];

        let mut cg = CompiledGraph::new(nodes, 1, 1, vec![]);
        cg.eval(&[4.0]).unwrap();
        let grad = cg.gradient();
        assert_relative_eq!(grad[0], 8.0, epsilon = 1e-12); // df/dx = 2*4 = 8
    }

    #[test]
    fn test_gradient_constant_function() {
        // f(x) = 5.0, df/dx = 0
        let nodes = vec![
            NodeOp::Input(0),      // node 0: x
            NodeOp::Constant(5.0), // node 1: 5
        ];

        let mut cg = CompiledGraph::new(nodes, 1, 1, vec![]);
        cg.eval(&[3.0]).unwrap();
        let grad = cg.gradient();
        assert_relative_eq!(grad[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gradient_identity() {
        // f(x) = x, df/dx = 1
        let nodes = vec![
            NodeOp::Input(0), // node 0: x
        ];

        let mut cg = CompiledGraph::new(nodes, 1, 0, vec![]);
        cg.eval(&[7.0]).unwrap();
        let grad = cg.gradient();
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gradient_deep_chain() {
        // f(x) = sin(exp(x)), df/dx = cos(exp(x)) * exp(x)
        let nodes = vec![
            NodeOp::Input(0), // node 0: x
            NodeOp::Unary {
                op: UnaryOp::Exp,
                src: 0,
            }, // node 1: exp(x)
            NodeOp::Unary {
                op: UnaryOp::Sin,
                src: 1,
            }, // node 2: sin(exp(x))
        ];

        let mut cg = CompiledGraph::new(nodes, 1, 2, vec![]);
        cg.eval(&[0.5]).unwrap();
        let e05 = 0.5_f64.exp();
        let expected = e05.cos() * e05;
        let grad = cg.gradient();
        assert_relative_eq!(grad[0], expected, epsilon = 1e-12);
    }

    #[test]
    fn test_eval_gradient_convenience() {
        // Same as test_gradient_two_inputs but using eval_gradient
        let nodes = vec![
            NodeOp::Input(0), // node 0: x
            NodeOp::Input(1), // node 1: y
            NodeOp::Binary {
                op: BinaryOp::Mul,
                lhs: 0,
                rhs: 1,
            },
        ];

        let mut cg = CompiledGraph::new(nodes, 2, 2, vec![]);
        let grad = cg.eval_gradient(&[3.0, 5.0]).unwrap();
        assert_relative_eq!(grad[0], 5.0, epsilon = 1e-12);
        assert_relative_eq!(grad[1], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gradient_multi_point() {
        // f(x, y) = x * y at multiple evaluation points
        let nodes = vec![
            NodeOp::Input(0),
            NodeOp::Input(1),
            NodeOp::Binary {
                op: BinaryOp::Mul,
                lhs: 0,
                rhs: 1,
            },
        ];

        let mut cg = CompiledGraph::new(nodes, 2, 2, vec![]);

        for &(x, y) in &[(1.0, 2.0), (3.0, 4.0), (-1.0, 5.0), (0.0, 7.0)] {
            let grad = cg.eval_gradient(&[x, y]).unwrap();
            assert_relative_eq!(grad[0], y, epsilon = 1e-12);
            assert_relative_eq!(grad[1], x, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_gradient_of_intermediate() {
        // Nodes: x, y, x*y, (x*y)^2
        // gradient_of node 2 (x*y) should give [y, x]
        let nodes = vec![
            NodeOp::Input(0), // node 0: x
            NodeOp::Input(1), // node 1: y
            NodeOp::Binary {
                op: BinaryOp::Mul,
                lhs: 0,
                rhs: 1,
            }, // node 2: x*y
            NodeOp::Binary {
                op: BinaryOp::Mul,
                lhs: 2,
                rhs: 2,
            }, // node 3: (x*y)^2
        ];

        let mut cg = CompiledGraph::new(nodes, 2, 3, vec![]);
        cg.eval(&[3.0, 5.0]).unwrap();

        // gradient of the intermediate node 2 (x*y), not the output
        let grad = cg.gradient_of(2);
        assert_relative_eq!(grad[0], 5.0, epsilon = 1e-12); // d(x*y)/dx = y
        assert_relative_eq!(grad[1], 3.0, epsilon = 1e-12); // d(x*y)/dy = x
    }

    #[test]
    fn test_compiled_graph_clone_independent() {
        // Clone produces independent evaluation state
        let mut ad = crate::AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let y = ad.var(0.0).unwrap();
        let x2 = ad.square(x);
        let xy = ad.mul(x, y);
        let f = ad.add(x2, xy);
        let template = ad.compile_primal(f, &[x, y]).unwrap();

        let mut cg1 = template.clone();
        let mut cg2 = template.clone();

        cg1.eval(&[1.0, 2.0]).unwrap();
        cg2.eval(&[3.0, 4.0]).unwrap();

        // Values are independent
        assert_relative_eq!(cg1.value(), 3.0, epsilon = 1e-12); // 1 + 2
        assert_relative_eq!(cg2.value(), 21.0, epsilon = 1e-12); // 9 + 12
    }

    #[test]
    fn test_compiled_graph_bevy_trait_bounds() {
        // Compile-time assertion: CompiledGraph can be used as Component and Resource
        fn assert_component<T: bevy_ecs::component::Component>() {}
        fn assert_resource<T: bevy_ecs::resource::Resource>() {}
        fn assert_clone<T: Clone>() {}
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_component::<CompiledGraph>();
        assert_resource::<CompiledGraph>();
        assert_clone::<CompiledGraph>();
        assert_send::<CompiledGraph>();
        assert_sync::<CompiledGraph>();
    }
}
