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

use crate::components::{BinaryInputs, UnaryInput};
use crate::components::{BinaryOp, IsConstant, IsInput, UnaryOp, Value};
use crate::context::{BinaryOpMarker, UnaryOpMarker};

/// A node in the flattened computation graph.
#[derive(Clone, Copy, Debug)]
pub enum NodeOp {
    /// Read from inputs[index].
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
/// Created by [`AutoDiff::compile`] or [`AutoDiff::compile_order`].
/// Stores a flattened node array for the function value and all
/// requested partial derivatives, enabling fast forward-pass evaluation
/// without ECS overhead.
pub struct CompiledGraph {
    nodes: Vec<NodeOp>,
    num_inputs: usize,
    output_index: usize,
    partial_outputs: Vec<(Vec<usize>, usize)>,
    values: Vec<f64>,
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
        Self {
            nodes,
            num_inputs,
            output_index,
            partial_outputs,
            values: vec![0.0; num_nodes],
        }
    }

    /// Evaluates the compiled graph at the given input values.
    ///
    /// After calling this, use `value()` and `partial()` to read results.
    pub fn eval(&mut self, inputs: &[f64]) {
        assert_eq!(
            inputs.len(),
            self.num_inputs,
            "expected {} inputs, got {}",
            self.num_inputs,
            inputs.len()
        );

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
    }

    /// Returns the function value from the most recent `eval()`.
    #[inline]
    pub fn value(&self) -> f64 {
        self.values[self.output_index]
    }

    /// Returns a partial derivative from the most recent `eval()`.
    ///
    /// The `multi_index` must match one of the partials passed to `compile()`.
    ///
    /// # Panics
    /// Panics if the requested partial was not compiled.
    pub fn partial(&self, multi_index: &[usize]) -> f64 {
        for (mi, idx) in &self.partial_outputs {
            if mi.as_slice() == multi_index {
                return self.values[*idx];
            }
        }
        panic!(
            "Partial {:?} was not compiled. Available: {:?}",
            multi_index,
            self.partial_outputs
                .iter()
                .map(|(mi, _)| mi.clone())
                .collect::<Vec<_>>()
        );
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

    /// Returns all available partial multi-indices.
    pub fn available_partials(&self) -> Vec<Vec<usize>> {
        self.partial_outputs
            .iter()
            .map(|(mi, _)| mi.clone())
            .collect()
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
                let val = entity_ref.get::<Value>().unwrap().get();
                nodes.push(NodeOp::Constant(val));
            }
        } else if entity_ref.contains::<IsConstant>() {
            let val = entity_ref.get::<Value>().unwrap().get();
            nodes.push(NodeOp::Constant(val));
        } else if let Some(&UnaryOpMarker(op)) = entity_ref.get::<UnaryOpMarker>() {
            let src_entity = entity_ref.get::<UnaryInput>().unwrap().get().entity();
            let src = entity_to_index[&src_entity];
            nodes.push(NodeOp::Unary { op, src });
        } else if let Some(&BinaryOpMarker(op)) = entity_ref.get::<BinaryOpMarker>() {
            let binary = entity_ref.get::<BinaryInputs>().unwrap();
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
pub fn generate_multi_indices(n: usize, max_order: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = vec![0usize; n];
    generate_helper(n, max_order, 0, &mut current, &mut result);
    result
}

fn generate_helper(
    n: usize,
    max_total: usize,
    pos: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if pos == n {
        let total: usize = current.iter().sum();
        if total > 0 && total <= max_total {
            result.push(current.clone());
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
pub fn apply_unary_value(op: UnaryOp, x: f64) -> f64 {
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
pub fn apply_binary_value(op: BinaryOp, x: f64, y: f64) -> f64 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
        BinaryOp::Pow => x.powf(y),
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
        cg.eval(&[3.0]);
        assert_eq!(cg.value(), 7.0); // 2*3 + 1
        cg.eval(&[5.0]);
        assert_eq!(cg.value(), 11.0); // 2*5 + 1
    }
}
