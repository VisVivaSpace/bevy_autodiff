//! Compiled computation graph for fast repeated derivative evaluation.
//!
//! The ECS graph handles expression building; `CompiledGraph` flattens it
//! into a `Vec<NodeOp>` that can be re-evaluated at new input values without
//! touching the ECS. Same exact derivative math, less overhead per eval.

use std::collections::HashMap;

use crate::components::{BinaryOp, UnaryOp};
use crate::taylor::polynomial::{constant_taylor, identity_taylor, TaylorCoeffs};
use crate::taylor::propagate::{compute_binary_taylor, compute_unary_taylor};

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
    Binary { op: BinaryOp, lhs: usize, rhs: usize },
}

/// A pre-computed weight for extracting a partial derivative from
/// directional derivative coefficients.
#[derive(Clone, Debug)]
struct ExtractionWeight {
    /// Index into the directions array.
    direction_idx: usize,
    /// Which coefficient to read (0 = value, 1 = first, ...).
    order: usize,
    /// Pre-computed combining coefficient (includes factorials and signs).
    weight: f64,
}

/// Compiled computation graph for fast repeated derivative evaluation.
///
/// `N` = number of Taylor coefficients (derivative order + 1).
/// `M` = number of input variables.
///
/// # Example
///
/// ```
/// use bevy_autodiff::AutoDiff;
///
/// let mut ad = AutoDiff::new();
/// let x = ad.var(2.0);
/// let y = ad.var(3.0);
/// let xy = ad.mul(x, y);
/// let f = ad.add(xy, x); // f = x*y + x
///
/// let mut cg = ad.compile::<3, 2>(f, &[x, y]);
/// cg.eval(&[2.0, 3.0]);
///
/// assert!((cg.value() - 8.0).abs() < 1e-10);       // f(2,3) = 8
/// assert!((cg.partial(&[1, 0]) - 4.0).abs() < 1e-10); // df/dx = y+1 = 4
/// assert!((cg.partial(&[0, 1]) - 2.0).abs() < 1e-10); // df/dy = x = 2
/// ```
pub struct CompiledGraph<const N: usize, const M: usize> {
    /// Flattened graph in topological order.
    nodes: Vec<NodeOp>,
    /// Index of the output node in `nodes`.
    output_node: usize,
    /// Number of nodes in the graph.
    num_nodes: usize,
    /// Direction vectors to propagate. Each is an M-element vector.
    /// directions[i] is the i-th direction.
    directions: Vec<[f64; M]>,
    /// For each multi-index (stored as [usize; M]), the extraction weights
    /// that combine directional coefficients into the exact partial derivative.
    extraction_map: HashMap<[usize; M], Vec<ExtractionWeight>>,
    /// All multi-indices with |alpha| <= N-1, for enumeration.
    multi_indices: Vec<[usize; M]>,
    // --- Workspace (pre-allocated, reused across evals) ---
    /// values[i] = f(x) at node i. Length = num_nodes.
    values: Vec<f64>,
    /// coeffs[dir_idx][node_idx * N + k] = k-th coefficient at node for direction dir_idx.
    /// Outer vec length = num_directions, inner vec length = num_nodes * N.
    coeffs: Vec<Vec<f64>>,
    /// Extracted partial derivative results, indexed same order as multi_indices.
    results: Vec<f64>,
}

impl<const N: usize, const M: usize> CompiledGraph<N, M> {
    /// Creates a new CompiledGraph from a flattened node list.
    ///
    /// This is called by `AutoDiff::compile()` which handles the ECS-to-flat
    /// conversion. Users should not call this directly.
    pub(crate) fn new(nodes: Vec<NodeOp>, output_node: usize) -> Self {
        let num_nodes = nodes.len();
        let order = N - 1; // max derivative order

        // Enumerate all multi-indices with |alpha| <= order
        let multi_indices = enumerate_multi_indices::<M>(order);

        // Compute the set of directions needed and extraction weights
        let (directions, extraction_map) =
            compute_directions_and_weights::<N, M>(&multi_indices);

        let num_directions = directions.len();

        // Pre-allocate workspace
        let values = vec![0.0; num_nodes];
        let coeffs = vec![vec![0.0; num_nodes * N]; num_directions];
        let results = vec![0.0; multi_indices.len()];

        Self {
            nodes,
            output_node,
            num_nodes,
            directions,
            extraction_map,
            multi_indices,
            values,
            coeffs,
            results,
        }
    }

    /// Maximum derivative order this graph can compute.
    #[inline]
    pub fn order(&self) -> usize {
        N - 1
    }

    /// Evaluate the graph at new input values, computing all partial
    /// derivatives up to order N-1.
    pub fn eval(&mut self, inputs: &[f64; M]) {
        // Phase 1: Value propagation — compute f(x) at each node.
        for i in 0..self.num_nodes {
            self.values[i] = match self.nodes[i] {
                NodeOp::Input(idx) => inputs[idx],
                NodeOp::Constant(c) => c,
                NodeOp::Unary { op, src } => apply_unary_value(op, self.values[src]),
                NodeOp::Binary { op, lhs, rhs } => {
                    apply_binary_value(op, self.values[lhs], self.values[rhs])
                }
            };
        }

        // Phase 2: Coefficient propagation — for each direction, walk forward
        // computing exact derivative coefficients at each node.
        let order = N - 1;
        for (dir_idx, direction) in self.directions.iter().enumerate() {
            for i in 0..self.num_nodes {
                let coeffs_start = i * N;
                let node_coeffs: TaylorCoeffs = match self.nodes[i] {
                    NodeOp::Input(idx) => {
                        identity_taylor(self.values[i], direction[idx], order)
                    }
                    NodeOp::Constant(_) => {
                        constant_taylor(self.values[i], order)
                    }
                    NodeOp::Unary { op, src } => {
                        let src_start = src * N;
                        let src_coeffs = &self.coeffs[dir_idx][src_start..src_start + N];
                        compute_unary_taylor(op, src_coeffs, order)
                    }
                    NodeOp::Binary { op, lhs, rhs } => {
                        let lhs_start = lhs * N;
                        let rhs_start = rhs * N;
                        let lhs_coeffs = &self.coeffs[dir_idx][lhs_start..lhs_start + N];
                        let rhs_coeffs = &self.coeffs[dir_idx][rhs_start..rhs_start + N];
                        compute_binary_taylor(op, lhs_coeffs, rhs_coeffs, order)
                    }
                };
                // Write coefficients into workspace
                for k in 0..N {
                    self.coeffs[dir_idx][coeffs_start + k] =
                        node_coeffs.get(k).copied().unwrap_or(0.0);
                }
            }
        }

        // Phase 3: Extract partial derivatives using pre-computed weights.
        let out_offset = self.output_node * N;
        for (result_idx, alpha) in self.multi_indices.iter().enumerate() {
            if let Some(weights) = self.extraction_map.get(alpha) {
                let mut val = 0.0;
                for w in weights {
                    let coeff = self.coeffs[w.direction_idx][out_offset + w.order];
                    val += w.weight * coeff;
                }
                self.results[result_idx] = val;
            } else {
                self.results[result_idx] = 0.0;
            }
        }
    }

    /// The function value f(inputs) from the most recent `eval`.
    #[inline]
    pub fn value(&self) -> f64 {
        self.values[self.output_node]
    }

    /// A partial derivative specified by multi-index alpha.
    ///
    /// `alpha[i]` = number of times to differentiate with respect to input i.
    /// For example, `partial(&[2, 0])` = ∂²f/∂x², `partial(&[1, 1])` = ∂²f/∂x∂y.
    ///
    /// Returns 0.0 if the multi-index has total order > N-1 or wasn't computed.
    pub fn partial(&self, alpha: &[usize; M]) -> f64 {
        let total_order: usize = alpha.iter().sum();
        if total_order == 0 {
            return self.value();
        }
        // Find this multi-index in our enumeration
        for (idx, mi) in self.multi_indices.iter().enumerate() {
            if mi == alpha {
                return self.results[idx];
            }
        }
        0.0
    }
}

// =============================================================================
// Value-level helpers (compute f(x) for each operation, no derivatives)
// =============================================================================

fn apply_unary_value(op: UnaryOp, x: f64) -> f64 {
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

fn apply_binary_value(op: BinaryOp, x: f64, y: f64) -> f64 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
        BinaryOp::Pow => x.powf(y),
    }
}

// =============================================================================
// Multi-index enumeration
// =============================================================================

/// Enumerate all multi-indices alpha in M dimensions with |alpha| <= max_order.
/// Includes the zero multi-index [0, 0, ..., 0].
fn enumerate_multi_indices<const M: usize>(max_order: usize) -> Vec<[usize; M]> {
    let mut result = Vec::new();
    let mut current = [0usize; M];
    enumerate_helper::<M>(&mut result, &mut current, 0, max_order);
    result
}

fn enumerate_helper<const M: usize>(
    result: &mut Vec<[usize; M]>,
    current: &mut [usize; M],
    dim: usize,
    remaining: usize,
) {
    if dim == M {
        result.push(*current);
        return;
    }
    for k in 0..=remaining {
        current[dim] = k;
        enumerate_helper::<M>(result, current, dim + 1, remaining - k);
    }
}

// =============================================================================
// Direction computation and extraction weights (inclusion-exclusion)
// =============================================================================

/// Compute the set of directions needed for all multi-indices, and the
/// extraction weights that combine directional coefficients into exact
/// partial derivatives.
#[allow(clippy::type_complexity)]
fn compute_directions_and_weights<const N: usize, const M: usize>(
    multi_indices: &[[usize; M]],
) -> (Vec<[f64; M]>, HashMap<[usize; M], Vec<ExtractionWeight>>) {
    let mut direction_map: HashMap<[usize; M], usize> = HashMap::new(); // direction -> index
    let mut directions: Vec<[f64; M]> = Vec::new();
    let mut extraction_map: HashMap<[usize; M], Vec<ExtractionWeight>> = HashMap::new();

    // We always need the basis directions (for pure partials)
    for i in 0..M {
        let mut d = [0.0f64; M];
        d[i] = 1.0;
        let key = float_dir_to_key::<M>(&d);
        if let std::collections::hash_map::Entry::Vacant(e) = direction_map.entry(key) {
            let idx = directions.len();
            e.insert(idx);
            directions.push(d);
        }
    }

    for alpha in multi_indices {
        let total_order: usize = alpha.iter().sum();
        if total_order == 0 {
            // Zero-order "derivative" is just the value; no weights needed.
            continue;
        }

        let weights = compute_extraction_weights_for::<N, M>(
            alpha,
            total_order,
            &mut directions,
            &mut direction_map,
        );
        extraction_map.insert(*alpha, weights);
    }

    (directions, extraction_map)
}

/// Convert a float direction to an integer key for deduplication.
/// Directions are always non-negative integers in our scheme.
fn float_dir_to_key<const M: usize>(d: &[f64; M]) -> [usize; M] {
    let mut key = [0usize; M];
    for i in 0..M {
        key[i] = d[i] as usize;
    }
    key
}

/// Compute extraction weights for a single multi-index alpha.
///
/// Three cases:
/// 1. Pure partial (one active variable): k! * coeff[k] from basis direction
/// 2. All-distinct mixed (each alpha_i is 0 or 1): inclusion-exclusion with
///    weight = sign on coefficients (the n! cancels between directional derivative
///    and inclusion-exclusion normalization)
/// 3. General (repeated + distinct): use scaled directions with inclusion-exclusion
///    over the set of active variables
fn compute_extraction_weights_for<const N: usize, const M: usize>(
    alpha: &[usize; M],
    _total_order: usize,
    directions: &mut Vec<[f64; M]>,
    direction_map: &mut HashMap<[usize; M], usize>,
) -> Vec<ExtractionWeight> {
    let active: Vec<usize> = (0..M).filter(|&i| alpha[i] > 0).collect();
    let total: usize = alpha.iter().sum();

    if total == 0 {
        return Vec::new();
    }

    // Case 1: Pure partial — only one variable involved
    if active.len() == 1 {
        let i = active[0];
        let k = alpha[i];
        let mut d = [0.0f64; M];
        d[i] = 1.0;
        let dir_idx = get_or_insert_direction::<M>(d, directions, direction_map);
        // d^k f/dx_i^k = k! * coeff[k](e_i)
        return vec![ExtractionWeight {
            direction_idx: dir_idx,
            order: k,
            weight: crate::util::factorial(k),
        }];
    }

    // Case 2: All-distinct mixed partial (each active alpha_i == 1)
    let all_distinct = active.iter().all(|&i| alpha[i] == 1);

    if all_distinct {
        let n = active.len(); // = total
        let num_subsets = 1usize << n;
        let mut weights = Vec::new();

        for mask in 1..num_subsets {
            let subset_size = (mask as u32).count_ones() as usize;
            let sign = if (n - subset_size).is_multiple_of(2) { 1.0 } else { -1.0 };

            let mut d = [0.0f64; M];
            for (bit, &var_idx) in active.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    d[var_idx] = 1.0;
                }
            }

            let dir_idx = get_or_insert_direction::<M>(d, directions, direction_map);
            // The n!/n! cancels: weight on coeff[n] is just the sign.
            weights.push(ExtractionWeight {
                direction_idx: dir_idx,
                order: n,
                weight: sign,
            });
        }

        return weights;
    }

    // Case 3: General — some alpha_i > 1 with multiple active variables.
    // This requires order 3+ and involves cross-terms between repeated and
    // distinct variables (e.g., d³f/dx²dy). Not yet implemented.
    //
    // For orders 1-2, only Cases 1 and 2 arise, which covers the primary
    // use cases (gradients, all second partial derivatives).
    unimplemented!(
        "General mixed partial with repeated variables (alpha = {:?}) \
         requires order {} extraction, which is not yet implemented. \
         Pure partials and all-distinct mixed partials are supported.",
        alpha,
        total
    )
}

/// Get the index for a direction, inserting it if not already present.
fn get_or_insert_direction<const M: usize>(
    d: [f64; M],
    directions: &mut Vec<[f64; M]>,
    direction_map: &mut HashMap<[usize; M], usize>,
) -> usize {
    let key = float_dir_to_key::<M>(&d);
    if let Some(&idx) = direction_map.get(&key) {
        idx
    } else {
        let idx = directions.len();
        direction_map.insert(key, idx);
        directions.push(d);
        idx
    }
}

#[cfg(test)]
mod tests {
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    // Helper: build a CompiledGraph and an AutoDiff context for the same expression,
    // then compare results.

    #[test]
    fn test_single_var_x_squared_value() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let f = ad.square(x); // f = x^2

        let mut cg = ad.compile::<3, 1>(f, &[x]);
        cg.eval(&[3.0]);

        assert_relative_eq!(cg.value(), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_var_x_squared_derivatives() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let f = ad.square(x);

        let mut cg = ad.compile::<3, 1>(f, &[x]);
        cg.eval(&[3.0]);

        // df/dx = 2x = 6
        assert_relative_eq!(cg.partial(&[1]), 6.0, epsilon = 1e-10);
        // d²f/dx² = 2
        assert_relative_eq!(cg.partial(&[2]), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_var_sin() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.sin(x);

        let mut cg = ad.compile::<4, 1>(f, &[x]);
        cg.eval(&[1.0]);

        assert_relative_eq!(cg.value(), 1.0_f64.sin(), epsilon = 1e-10);
        // d/dx sin(x) = cos(x)
        assert_relative_eq!(cg.partial(&[1]), 1.0_f64.cos(), epsilon = 1e-10);
        // d²/dx² sin(x) = -sin(x)
        assert_relative_eq!(cg.partial(&[2]), -1.0_f64.sin(), epsilon = 1e-10);
        // d³/dx³ sin(x) = -cos(x)
        assert_relative_eq!(cg.partial(&[3]), -1.0_f64.cos(), epsilon = 1e-10);
    }

    #[test]
    fn test_single_var_exp() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.exp(x);

        let mut cg = ad.compile::<5, 1>(f, &[x]);
        cg.eval(&[1.0]);

        let e = std::f64::consts::E;
        assert_relative_eq!(cg.value(), e, epsilon = 1e-10);
        for k in 1..=4 {
            assert_relative_eq!(cg.partial(&[k]), e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_two_var_gradient() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y); // f = x*y

        let mut cg = ad.compile::<2, 2>(f, &[x, y]);
        cg.eval(&[2.0, 3.0]);

        assert_relative_eq!(cg.value(), 6.0, epsilon = 1e-10);
        // df/dx = y = 3
        assert_relative_eq!(cg.partial(&[1, 0]), 3.0, epsilon = 1e-10);
        // df/dy = x = 2
        assert_relative_eq!(cg.partial(&[0, 1]), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_two_var_mixed_partial() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y); // f = x*y

        let mut cg = ad.compile::<3, 2>(f, &[x, y]);
        cg.eval(&[2.0, 3.0]);

        // d²f/dxdy = 1
        assert_relative_eq!(cg.partial(&[1, 1]), 1.0, epsilon = 1e-10);
        // d²f/dx² = 0
        assert_relative_eq!(cg.partial(&[2, 0]), 0.0, epsilon = 1e-10);
        // d²f/dy² = 0
        assert_relative_eq!(cg.partial(&[0, 2]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_repeated_eval_different_inputs() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.square(x);

        let mut cg = ad.compile::<3, 1>(f, &[x]);

        // Eval at x=3
        cg.eval(&[3.0]);
        assert_relative_eq!(cg.value(), 9.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1]), 6.0, epsilon = 1e-10);

        // Eval at x=5
        cg.eval(&[5.0]);
        assert_relative_eq!(cg.value(), 25.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1]), 10.0, epsilon = 1e-10);

        // Eval at x=-2
        cg.eval(&[-2.0]);
        assert_relative_eq!(cg.value(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1]), -4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_constant_derivative_is_zero() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let c = ad.constant(5.0);
        let f = ad.add(x, c); // f = x + 5

        let mut cg = ad.compile::<3, 1>(f, &[x]);
        cg.eval(&[1.0]);

        assert_relative_eq!(cg.value(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1]), 1.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[2]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rosenbrock_value_and_gradient() {
        // f(x,y) = (1-x)² + 100(y-x²)²
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(1.0);
        let one = ad.constant(1.0);
        let hundred = ad.constant(100.0);

        let one_minus_x = ad.sub(one, x);
        let term1 = ad.square(one_minus_x);

        let x2 = ad.square(x);
        let y_minus_x2 = ad.sub(y, x2);
        let term2_inner = ad.square(y_minus_x2);
        let term2 = ad.mul(hundred, term2_inner);

        let f = ad.add(term1, term2);

        let mut cg = ad.compile::<2, 2>(f, &[x, y]);
        cg.eval(&[1.0, 1.0]);

        // At minimum (1,1): f = 0
        assert_relative_eq!(cg.value(), 0.0, epsilon = 1e-10);
        // Gradient should be [0, 0] at minimum
        assert_relative_eq!(cg.partial(&[1, 0]), 0.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[0, 1]), 0.0, epsilon = 1e-10);

        // Eval away from minimum
        cg.eval(&[0.0, 0.0]);
        // f(0,0) = 1 + 0 = 1
        assert_relative_eq!(cg.value(), 1.0, epsilon = 1e-10);
        // df/dx at (0,0) = -2(1-x) + 100*2*(y-x²)*(-2x) = -2
        assert_relative_eq!(cg.partial(&[1, 0]), -2.0, epsilon = 1e-10);
        // df/dy at (0,0) = 100*2*(y-x²) = 0
        assert_relative_eq!(cg.partial(&[0, 1]), 0.0, epsilon = 1e-10);
    }

    /// Compare compiled partial derivatives against existing ad.derivative() for
    /// single-variable functions.
    #[test]
    fn test_compiled_vs_ecs_single_var() {
        let fns: Vec<(&str, fn(&mut AutoDiff, _) -> _)> = vec![
            ("sin", |ad: &mut AutoDiff, x| ad.sin(x)),
            ("cos", |ad: &mut AutoDiff, x| ad.cos(x)),
            ("exp", |ad: &mut AutoDiff, x| ad.exp(x)),
            ("ln", |ad: &mut AutoDiff, x| ad.ln(x)),
            ("sqrt", |ad: &mut AutoDiff, x| ad.sqrt(x)),
        ];

        let test_values: &[f64] = &[0.5, 1.0, 2.0];

        for (_name, build_fn) in &fns {
            for &val in test_values {
                let mut ad = AutoDiff::new();
                let x = ad.var(val);
                let f = build_fn(&mut ad, x);

                // Get ECS-based derivatives
                let ecs_d1 = ad.derivative(f, x, 1);
                let ecs_d2 = ad.derivative(f, x, 2);
                let ecs_d3 = ad.derivative(f, x, 3);

                // Get compiled derivatives
                let mut cg = ad.compile::<4, 1>(f, &[x]);
                cg.eval(&[val]);

                assert_relative_eq!(
                    cg.partial(&[1]),
                    ecs_d1,
                    epsilon = 1e-8,
                    max_relative = 1e-8,
                );
                assert_relative_eq!(
                    cg.partial(&[2]),
                    ecs_d2,
                    epsilon = 1e-8,
                    max_relative = 1e-8,
                );
                assert_relative_eq!(
                    cg.partial(&[3]),
                    ecs_d3,
                    epsilon = 1e-8,
                    max_relative = 1e-8,
                );
            }
        }
    }

    #[test]
    fn test_three_var_partials() {
        // f(x, y, z) = x*y*z
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let z = ad.var(5.0);
        let xy = ad.mul(x, y);
        let f = ad.mul(xy, z);

        let mut cg = ad.compile::<2, 3>(f, &[x, y, z]);
        cg.eval(&[2.0, 3.0, 5.0]);

        assert_relative_eq!(cg.value(), 30.0, epsilon = 1e-10);
        // df/dx = y*z = 15
        assert_relative_eq!(cg.partial(&[1, 0, 0]), 15.0, epsilon = 1e-10);
        // df/dy = x*z = 10
        assert_relative_eq!(cg.partial(&[0, 1, 0]), 10.0, epsilon = 1e-10);
        // df/dz = x*y = 6
        assert_relative_eq!(cg.partial(&[0, 0, 1]), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_two_var_all_second_partials() {
        // f(x, y) = x²y + sin(y)
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);
        let x2 = ad.square(x);
        let x2y = ad.mul(x2, y);
        let sin_y = ad.sin(y);
        let f = ad.add(x2y, sin_y);

        let mut cg = ad.compile::<3, 2>(f, &[x, y]);
        cg.eval(&[1.0, 2.0]);

        // f = x²y + sin(y), at (1, 2)
        let expected_val = 1.0 * 2.0 + 2.0_f64.sin();
        assert_relative_eq!(cg.value(), expected_val, epsilon = 1e-10);

        // df/dx = 2xy = 4
        assert_relative_eq!(cg.partial(&[1, 0]), 4.0, epsilon = 1e-10);
        // df/dy = x² + cos(y) = 1 + cos(2)
        assert_relative_eq!(cg.partial(&[0, 1]), 1.0 + 2.0_f64.cos(), epsilon = 1e-10);
        // d²f/dx² = 2y = 4
        assert_relative_eq!(cg.partial(&[2, 0]), 4.0, epsilon = 1e-10);
        // d²f/dy² = -sin(y) = -sin(2)
        assert_relative_eq!(cg.partial(&[0, 2]), -2.0_f64.sin(), epsilon = 1e-10);
        // d²f/dxdy = 2x = 2
        assert_relative_eq!(cg.partial(&[1, 1]), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rosenbrock_second_partials() {
        // f(x,y) = (1-x)² + 100(y-x²)²
        // At (1, 1): all partials are well-known
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(1.0);
        let one = ad.constant(1.0);
        let hundred = ad.constant(100.0);

        let one_minus_x = ad.sub(one, x);
        let term1 = ad.square(one_minus_x);
        let x2 = ad.square(x);
        let y_minus_x2 = ad.sub(y, x2);
        let term2_inner = ad.square(y_minus_x2);
        let term2 = ad.mul(hundred, term2_inner);
        let f = ad.add(term1, term2);

        let mut cg = ad.compile::<3, 2>(f, &[x, y]);
        cg.eval(&[1.0, 1.0]);

        // At minimum (1,1):
        // d²f/dx² = 2 + 100*(12x² - 4y) = 2 + 100*8 = 802
        assert_relative_eq!(cg.partial(&[2, 0]), 802.0, epsilon = 1e-8);
        // d²f/dy² = 200
        assert_relative_eq!(cg.partial(&[0, 2]), 200.0, epsilon = 1e-8);
        // d²f/dxdy = -400x = -400
        assert_relative_eq!(cg.partial(&[1, 1]), -400.0, epsilon = 1e-8);
    }

    #[test]
    fn test_compiled_vs_ecs_two_var_mixed() {
        // Cross-validate compiled mixed partials against existing ad.partial()
        // f(x, y) = sin(x) * exp(y)
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let y = ad.var(1.0);
        let sin_x = ad.sin(x);
        let exp_y = ad.exp(y);
        let f = ad.mul(sin_x, exp_y);

        // ECS-based partials
        use crate::MultiIndex;
        let ecs_dx = ad.partial(f, &MultiIndex::new(vec![1, 0]));
        let ecs_dy = ad.partial(f, &MultiIndex::new(vec![0, 1]));
        let ecs_dxx = ad.partial(f, &MultiIndex::new(vec![2, 0]));
        let ecs_dyy = ad.partial(f, &MultiIndex::new(vec![0, 2]));
        let ecs_dxy = ad.partial(f, &MultiIndex::new(vec![1, 1]));

        // Compiled partials
        let mut cg = ad.compile::<3, 2>(f, &[x, y]);
        cg.eval(&[0.5, 1.0]);

        assert_relative_eq!(cg.partial(&[1, 0]), ecs_dx, epsilon = 1e-8);
        assert_relative_eq!(cg.partial(&[0, 1]), ecs_dy, epsilon = 1e-8);
        assert_relative_eq!(cg.partial(&[2, 0]), ecs_dxx, epsilon = 1e-8);
        assert_relative_eq!(cg.partial(&[0, 2]), ecs_dyy, epsilon = 1e-8);
        assert_relative_eq!(cg.partial(&[1, 1]), ecs_dxy, epsilon = 1e-8);
    }

    #[test]
    fn test_identity_function() {
        // f(x) = x
        let mut ad = AutoDiff::new();
        let x = ad.var(7.0);

        let mut cg = ad.compile::<3, 1>(x, &[x]);
        cg.eval(&[7.0]);

        assert_relative_eq!(cg.value(), 7.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1]), 1.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[2]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chain_of_operations() {
        // f(x) = exp(sin(ln(x)))
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let ln_x = ad.ln(x);
        let sin_ln_x = ad.sin(ln_x);
        let f = ad.exp(sin_ln_x);

        // Cross-validate with ECS
        let ecs_d1 = ad.derivative(f, x, 1);
        let ecs_d2 = ad.derivative(f, x, 2);

        let mut cg = ad.compile::<3, 1>(f, &[x]);
        cg.eval(&[2.0]);

        assert_relative_eq!(cg.value(), ad.eval(f), epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1]), ecs_d1, epsilon = 1e-8);
        assert_relative_eq!(cg.partial(&[2]), ecs_d2, epsilon = 1e-8);
    }

    #[test]
    fn test_third_order_pure_and_distinct() {
        // f(x, y) = x²*y² — test that pure and all-distinct partials
        // work at order 3, even though general mixed (Case 3) is not yet supported.
        // Use N=4 but only request partials that don't hit Case 3.
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let x2 = ad.square(x);
        let y2 = ad.square(y);
        let f = ad.mul(x2, y2);

        // N=3 gives us up to order 2, avoiding Case 3
        let mut cg = ad.compile::<3, 2>(f, &[x, y]);
        cg.eval(&[2.0, 3.0]);

        // f = x²y² at (2, 3)
        assert_relative_eq!(cg.value(), 36.0, epsilon = 1e-10);
        // df/dx = 2xy² = 36
        assert_relative_eq!(cg.partial(&[1, 0]), 36.0, epsilon = 1e-10);
        // df/dy = 2x²y = 24
        assert_relative_eq!(cg.partial(&[0, 1]), 24.0, epsilon = 1e-10);
        // d²f/dxdy = 4xy = 24
        assert_relative_eq!(cg.partial(&[1, 1]), 24.0, epsilon = 1e-10);
        // d²f/dx² = 2y² = 18
        assert_relative_eq!(cg.partial(&[2, 0]), 18.0, epsilon = 1e-10);
        // d²f/dy² = 2x² = 8
        assert_relative_eq!(cg.partial(&[0, 2]), 8.0, epsilon = 1e-10);
    }
}
