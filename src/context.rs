//! AutoDiff context - the main API for building computation graphs.

use bevy_ecs::world::World;
use bevy_entity_ptr::EntityHandle;

use crate::components::{
    BinaryInputs, BinaryOp, Dependencies, IsConstant, IsInput, TaylorData, UnaryInput, UnaryOp,
    Value, Variable,
};
use crate::var::Var;

/// Component marker for unary operations (stores the operation type).
#[derive(bevy_ecs::component::Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnaryOpMarker(pub UnaryOp);

/// Component marker for binary operations (stores the operation type).
#[derive(bevy_ecs::component::Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryOpMarker(pub BinaryOp);

/// The main autodiff context for building and evaluating computation graphs.
///
/// `AutoDiff` owns the ECS world that stores the computation graph.
/// Variables are created as entities with components that define their
/// role in the graph.
///
/// # Example
///
/// ```
/// use bevy_autodiff::AutoDiff;
///
/// let mut ad = AutoDiff::new();
///
/// // Create input variables
/// let x = ad.var(2.0);
/// let y = ad.var(3.0);
///
/// // Build computation graph: f = x * y + x
/// let xy = ad.mul(x, y);
/// let f = ad.add(xy, x);
///
/// // Evaluate
/// assert_eq!(ad.eval(f), 8.0); // 2*3 + 2 = 8
/// ```
pub struct AutoDiff {
    /// The ECS world storing the computation graph.
    world: World,
    /// Counter for assigning input indices (for dependency tracking).
    input_count: usize,
}

impl AutoDiff {
    /// Creates a new, empty autodiff context.
    #[inline]
    pub fn new() -> Self {
        Self {
            world: World::new(),
            input_count: 0,
        }
    }

    /// Returns a reference to the underlying ECS world.
    /// Useful for advanced operations or debugging.
    #[inline]
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Returns a mutable reference to the underlying ECS world.
    /// Use with care - modifying the world directly can invalidate cached Taylor data.
    #[inline]
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Creates a new input variable with the given initial value.
    ///
    /// Input variables are the leaves of the computation graph.
    /// Derivatives are computed with respect to these inputs.
    pub fn var(&mut self, value: f64) -> Var {
        let input_index = self.input_count;
        self.input_count += 1;

        let entity = self
            .world
            .spawn((
                Variable,
                IsInput,
                Value::new(value),
                Dependencies::single(input_index),
                TaylorData::constant(value),
            ))
            .id();

        Var::new(entity)
    }

    /// Creates a constant variable with the given value.
    ///
    /// Constants have zero derivatives with respect to all inputs.
    /// They are useful for embedding fixed values in the computation graph.
    pub fn constant(&mut self, value: f64) -> Var {
        let entity = self
            .world
            .spawn((
                Variable,
                IsConstant,
                Value::new(value),
                Dependencies::none(),
                TaylorData::constant(value),
            ))
            .id();

        Var::new(entity)
    }

    /// Evaluates a variable, returning its current numerical value.
    ///
    /// For input variables, this returns the stored value.
    /// For computed variables, this returns the result of propagating
    /// values through the graph.
    pub fn eval(&self, var: Var) -> f64 {
        self.world
            .entity(var.entity())
            .get::<Value>()
            .map(|v| v.get())
            .expect("Variable missing Value component")
    }

    /// Sets the value of an input variable.
    ///
    /// This will invalidate any cached Taylor data that depends on this input.
    ///
    /// # Panics
    /// Panics if the variable is not an input variable.
    pub fn set_input(&mut self, var: Var, value: f64) {
        let entity = var.entity();

        // Verify it's an input
        if !self.world.entity(entity).contains::<IsInput>() {
            panic!("set_input called on non-input variable");
        }

        // Update value
        if let Some(mut val) = self.world.entity_mut(entity).get_mut::<Value>() {
            val.0 = value;
        }

        // Update Taylor data (constant for inputs)
        if let Some(mut taylor) = self.world.entity_mut(entity).get_mut::<TaylorData>() {
            *taylor = TaylorData::constant(value);
        }
    }

    // =========================================================================
    // Binary Operations
    // =========================================================================

    /// Creates a new variable representing the sum of two variables: a + b
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.binary_op(BinaryOp::Add, a, b, |x, y| x + y)
    }

    /// Creates a new variable representing the difference of two variables: a - b
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.binary_op(BinaryOp::Sub, a, b, |x, y| x - y)
    }

    /// Creates a new variable representing the product of two variables: a * b
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        self.binary_op(BinaryOp::Mul, a, b, |x, y| x * y)
    }

    /// Creates a new variable representing the quotient of two variables: a / b
    ///
    /// # Note
    /// Division by zero will produce NaN or infinity in the value.
    pub fn div(&mut self, a: Var, b: Var) -> Var {
        self.binary_op(BinaryOp::Div, a, b, |x, y| x / y)
    }

    /// Creates a new variable representing a raised to the power b: a^b
    pub fn pow(&mut self, base: Var, exponent: Var) -> Var {
        self.binary_op(BinaryOp::Pow, base, exponent, |x, y| x.powf(y))
    }

    /// Internal helper for creating binary operations.
    fn binary_op(&mut self, op: BinaryOp, a: Var, b: Var, f: fn(f64, f64) -> f64) -> Var {
        // Get values
        let a_val = self.eval(a);
        let b_val = self.eval(b);
        let result = f(a_val, b_val);

        // Compute dependencies (union of inputs)
        let a_deps = self
            .world
            .entity(a.entity())
            .get::<Dependencies>()
            .cloned()
            .unwrap_or_else(Dependencies::none);
        let b_deps = self
            .world
            .entity(b.entity())
            .get::<Dependencies>()
            .cloned()
            .unwrap_or_else(Dependencies::none);
        let deps = a_deps.union(&b_deps);

        let entity = self
            .world
            .spawn((
                Variable,
                Value::new(result),
                BinaryOpMarker(op),
                BinaryInputs::new(a.handle(), b.handle()),
                deps,
                TaylorData::constant(result),
            ))
            .id();

        Var::new(entity)
    }

    // =========================================================================
    // Unary Operations
    // =========================================================================

    /// Creates a new variable representing the negation: -x
    pub fn neg(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Neg, x, |v| -v)
    }

    /// Creates a new variable representing sin(x)
    pub fn sin(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Sin, x, f64::sin)
    }

    /// Creates a new variable representing cos(x)
    pub fn cos(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Cos, x, f64::cos)
    }

    /// Creates a new variable representing exp(x) = e^x
    pub fn exp(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Exp, x, f64::exp)
    }

    /// Creates a new variable representing ln(x)
    ///
    /// # Note
    /// Returns NaN for negative inputs, -infinity for zero.
    pub fn ln(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Ln, x, f64::ln)
    }

    /// Creates a new variable representing sqrt(x)
    ///
    /// # Note
    /// Returns NaN for negative inputs.
    pub fn sqrt(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Sqrt, x, f64::sqrt)
    }

    /// Creates a new variable representing sinh(x)
    pub fn sinh(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Sinh, x, f64::sinh)
    }

    /// Creates a new variable representing cosh(x)
    pub fn cosh(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Cosh, x, f64::cosh)
    }

    /// Creates a new variable representing tan(x)
    pub fn tan(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Tan, x, f64::tan)
    }

    /// Creates a new variable representing tanh(x)
    pub fn tanh(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Tanh, x, f64::tanh)
    }

    /// Creates a new variable representing asin(x)
    ///
    /// # Note
    /// Returns NaN for inputs outside [-1, 1].
    pub fn asin(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Asin, x, f64::asin)
    }

    /// Creates a new variable representing acos(x)
    ///
    /// # Note
    /// Returns NaN for inputs outside [-1, 1].
    pub fn acos(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Acos, x, f64::acos)
    }

    /// Creates a new variable representing atan(x)
    pub fn atan(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Atan, x, f64::atan)
    }

    /// Creates a new variable representing asinh(x)
    pub fn asinh(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Asinh, x, f64::asinh)
    }

    /// Creates a new variable representing acosh(x)
    ///
    /// # Note
    /// Returns NaN for inputs less than 1.
    pub fn acosh(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Acosh, x, f64::acosh)
    }

    /// Creates a new variable representing atanh(x)
    ///
    /// # Note
    /// Returns NaN for inputs outside (-1, 1).
    pub fn atanh(&mut self, x: Var) -> Var {
        self.unary_op(UnaryOp::Atanh, x, f64::atanh)
    }

    /// Internal helper for creating unary operations.
    fn unary_op(&mut self, op: UnaryOp, x: Var, f: fn(f64) -> f64) -> Var {
        let x_val = self.eval(x);
        let result = f(x_val);

        // Copy dependencies from input
        let deps = self
            .world
            .entity(x.entity())
            .get::<Dependencies>()
            .cloned()
            .unwrap_or_else(Dependencies::none);

        let entity = self
            .world
            .spawn((
                Variable,
                Value::new(result),
                UnaryOpMarker(op),
                UnaryInput::new(EntityHandle::new(x.entity())),
                deps,
                TaylorData::constant(result),
            ))
            .id();

        Var::new(entity)
    }

    // =========================================================================
    // Convenience Operations
    // =========================================================================

    /// Creates a new variable representing x^2 (more efficient than pow for squares)
    #[inline]
    pub fn square(&mut self, x: Var) -> Var {
        self.mul(x, x)
    }

    /// Creates a new variable representing x^n for integer n
    pub fn powi(&mut self, x: Var, n: i32) -> Var {
        let n_const = self.constant(n as f64);
        self.pow(x, n_const)
    }

    /// Creates a new variable representing x^p for float p
    pub fn powf(&mut self, x: Var, p: f64) -> Var {
        let p_const = self.constant(p);
        self.pow(x, p_const)
    }

    // =========================================================================
    // Graph Information
    // =========================================================================

    /// Returns the number of input variables in the graph.
    #[inline]
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Returns true if the variable is an input variable.
    #[inline]
    pub fn is_input(&self, var: Var) -> bool {
        self.world.entity(var.entity()).contains::<IsInput>()
    }

    /// Returns true if the variable is a constant.
    #[inline]
    pub fn is_constant(&self, var: Var) -> bool {
        self.world.entity(var.entity()).contains::<IsConstant>()
    }

    /// Returns true if variable `output` depends on variable `input`.
    pub fn depends_on(&self, output: Var, input: Var) -> bool {
        // Get input's index
        let input_deps = self.world.entity(input.entity()).get::<Dependencies>();

        // If input isn't tracked, check if they're the same entity
        if input_deps.map(|d| d.is_empty()).unwrap_or(true) && !self.is_input(input) {
            return output.entity() == input.entity();
        }

        // Find which bit this input uses
        let input_mask = input_deps.map(|d| d.mask).unwrap_or(0);

        // Check if output depends on any of those inputs
        let output_deps = self.world.entity(output.entity()).get::<Dependencies>();
        output_deps.map(|d| (d.mask & input_mask) != 0).unwrap_or(false)
    }

    // =========================================================================
    // Differentiation
    // =========================================================================

    /// Computes the k-th derivative of `output` with respect to `input`.
    ///
    /// Uses Taylor series propagation: parameterize along the direction
    /// from the current input value with unit change in the target input,
    /// then extract the k-th Taylor coefficient multiplied by k!.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_autodiff::AutoDiff;
    ///
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(2.0);
    /// let y = ad.square(x);  // y = x²
    ///
    /// // dy/dx = 2x = 4 at x=2
    /// assert_eq!(ad.derivative(y, x, 1), 4.0);
    ///
    /// // d²y/dx² = 2
    /// assert_eq!(ad.derivative(y, x, 2), 2.0);
    /// ```
    pub fn derivative(&mut self, output: Var, input: Var, order: usize) -> f64 {
        use crate::components::Direction;
        use crate::taylor::propagate::{extract_derivative, propagate_taylor};

        // Get the input's index in the dependency mask
        let input_deps = self
            .world
            .entity(input.entity())
            .get::<Dependencies>()
            .cloned()
            .unwrap_or_else(Dependencies::none);

        if input_deps.is_empty() {
            // If input has no dependencies tracked (constant), derivative is 0
            return 0.0;
        }

        let input_index = input_deps.mask.trailing_zeros() as usize;

        // Create a direction vector with 1 in the input's position
        let direction = Direction::basis(self.input_count, input_index);

        // Propagate Taylor coefficients
        let coeffs = propagate_taylor(&mut self.world, output.entity(), &direction, order);

        // Extract the k-th derivative
        extract_derivative(&coeffs, order)
    }

    /// Computes the gradient of `output` with respect to all inputs.
    ///
    /// Returns a vector of first derivatives, one for each input variable
    /// in the order they were created.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_autodiff::AutoDiff;
    ///
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(1.0);
    /// let y = ad.var(2.0);
    /// let x2 = ad.square(x);
    /// let y2 = ad.square(y);
    /// let f = ad.add(x2, y2);  // f = x² + y²
    ///
    /// let grad = ad.gradient(f);
    /// assert_eq!(grad, vec![2.0, 4.0]);  // [∂f/∂x, ∂f/∂y] = [2x, 2y]
    /// ```
    pub fn gradient(&mut self, output: Var) -> Vec<f64> {
        use crate::components::Direction;
        use crate::taylor::propagate::{extract_derivative, propagate_taylor};

        let mut grad = Vec::with_capacity(self.input_count);

        for i in 0..self.input_count {
            let direction = Direction::basis(self.input_count, i);
            let coeffs = propagate_taylor(&mut self.world, output.entity(), &direction, 1);
            grad.push(extract_derivative(&coeffs, 1));
        }

        grad
    }

    /// Computes a partial derivative specified by a multi-index.
    ///
    /// The multi-index α = (α₁, α₂, ..., αₙ) specifies:
    /// ∂^|α|f / ∂x₁^α₁ ∂x₂^α₂ ... ∂xₙ^αₙ
    ///
    /// where |α| = α₁ + α₂ + ... + αₙ is the total order.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_autodiff::{AutoDiff, MultiIndex};
    ///
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(2.0);
    /// let y = ad.var(3.0);
    /// let f = ad.mul(x, y);  // f = x * y
    ///
    /// // ∂²f/∂x∂y = 1
    /// let index = MultiIndex::new(vec![1, 1]);
    /// let mixed = ad.partial(f, &index);
    /// assert!((mixed - 1.0).abs() < 1e-10);
    /// ```
    pub fn partial(&mut self, output: Var, index: &crate::components::MultiIndex) -> f64 {
        crate::partials::compute_partial(&mut self.world, output, index, self.input_count)
    }

    /// Computes the gradient using reverse mode (backpropagation).
    ///
    /// This is more efficient than `gradient()` (forward mode) when there are
    /// many inputs and one output, as it only requires one backward pass
    /// instead of one forward pass per input.
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_autodiff::AutoDiff;
    ///
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(1.0);
    /// let y = ad.var(2.0);
    /// let x2 = ad.square(x);
    /// let y2 = ad.square(y);
    /// let f = ad.add(x2, y2);  // f = x² + y²
    ///
    /// // ∇f = [2x, 2y] = [2, 4]
    /// let grad = ad.gradient_reverse(f);
    /// assert_eq!(grad, vec![2.0, 4.0]);
    /// ```
    pub fn gradient_reverse(&mut self, output: Var) -> Vec<f64> {
        crate::reverse::compute_gradient_reverse(&mut self.world, output, self.input_count)
    }

    // =========================================================================
    // Higher-Order API (Phase 7)
    // =========================================================================

    /// Computes the Hessian matrix (matrix of second partial derivatives).
    ///
    /// Returns an n×n matrix H where H[i][j] = ∂²f/∂xᵢ∂xⱼ.
    /// The Hessian is symmetric for smooth functions (Schwarz's theorem).
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_autodiff::AutoDiff;
    ///
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(1.0);
    /// let y = ad.var(2.0);
    /// let x2 = ad.square(x);
    /// let y2 = ad.square(y);
    /// let f = ad.add(x2, y2);  // f = x² + y²
    ///
    /// let hess = ad.hessian(f);
    /// // H = [[2, 0], [0, 2]]
    /// assert_eq!(hess[0][0], 2.0);
    /// assert_eq!(hess[0][1], 0.0);
    /// assert_eq!(hess[1][0], 0.0);
    /// assert_eq!(hess[1][1], 2.0);
    /// ```
    pub fn hessian(&mut self, output: Var) -> Vec<Vec<f64>> {
        use crate::components::MultiIndex;

        let n = self.input_count;
        let mut hess = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                let mut index = vec![0; n];
                index[i] += 1;
                index[j] += 1;
                let multi = MultiIndex::new(index);

                let value = self.partial(output, &multi);
                hess[i][j] = value;
                hess[j][i] = value; // Symmetry
            }
        }

        hess
    }

    /// Computes the Jacobian matrix for multiple outputs.
    ///
    /// Returns an m×n matrix J where J[i][j] = ∂fᵢ/∂xⱼ.
    ///
    /// # Arguments
    /// - `outputs`: Vector of output variables
    ///
    /// # Example
    ///
    /// ```
    /// use bevy_autodiff::AutoDiff;
    ///
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(1.0);
    /// let y = ad.var(2.0);
    ///
    /// let f1 = ad.mul(x, y);        // f1 = x*y
    /// let f2 = ad.add(x, y);        // f2 = x + y
    ///
    /// let jac = ad.jacobian(&[f1, f2]);
    /// // J = [[∂f1/∂x, ∂f1/∂y], [∂f2/∂x, ∂f2/∂y]]
    /// //   = [[y, x], [1, 1]]
    /// //   = [[2, 1], [1, 1]]
    /// assert_eq!(jac[0], vec![2.0, 1.0]);
    /// assert_eq!(jac[1], vec![1.0, 1.0]);
    /// ```
    pub fn jacobian(&mut self, outputs: &[Var]) -> Vec<Vec<f64>> {
        outputs.iter().map(|&out| self.gradient(out)).collect()
    }

    /// Sets multiple input values at once.
    ///
    /// This is more efficient than calling `set_input` multiple times when
    /// you need to update several inputs, as it batches the cache invalidation.
    ///
    /// # Arguments
    /// - `inputs`: Slice of (Var, value) pairs
    ///
    /// # Panics
    /// Panics if any variable is not an input variable.
    pub fn set_inputs(&mut self, inputs: &[(Var, f64)]) {
        for &(var, value) in inputs {
            self.set_input(var, value);
        }
    }

    /// Compiles the computation graph for a given output and inputs into a
    /// flat representation that can be re-evaluated efficiently at new input
    /// values without touching the ECS.
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
    /// let f = ad.square(x); // f = x²
    ///
    /// let mut cg = ad.compile::<3, 1>(f, &[x]);
    /// cg.eval(&[3.0]);
    /// assert!((cg.partial(&[1]) - 6.0).abs() < 1e-10); // df/dx = 2x = 6
    /// ```
    pub fn compile<const N: usize, const M: usize>(
        &self,
        output: Var,
        inputs: &[Var; M],
    ) -> crate::compiled::CompiledGraph<N, M> {
        use crate::compiled::NodeOp;
        use crate::graph::topological_order;
        use std::collections::HashMap;

        let topo = topological_order(&self.world, output.entity());

        // Map each entity to a flat node index
        let mut entity_to_idx: HashMap<bevy_ecs::entity::Entity, usize> = HashMap::new();
        for (i, &entity) in topo.iter().enumerate() {
            entity_to_idx.insert(entity, i);
        }

        // Build input entity -> input index mapping
        let mut input_entity_to_idx: HashMap<bevy_ecs::entity::Entity, usize> = HashMap::new();
        for (i, var) in inputs.iter().enumerate() {
            input_entity_to_idx.insert(var.entity(), i);
        }

        // Convert each entity to a NodeOp
        let mut nodes = Vec::with_capacity(topo.len());
        for &entity in &topo {
            let entity_ref = self.world.entity(entity);

            let node = if let Some(&input_idx) = input_entity_to_idx.get(&entity) {
                NodeOp::Input(input_idx)
            } else if entity_ref.contains::<IsConstant>() {
                let value = entity_ref.get::<Value>().map(|v| v.get()).unwrap_or(0.0);
                NodeOp::Constant(value)
            } else if let Some(op_marker) = entity_ref.get::<crate::context::UnaryOpMarker>() {
                let input_handle = entity_ref
                    .get::<UnaryInput>()
                    .expect("UnaryOp missing input");
                let src = entity_to_idx[&input_handle.get().entity()];
                NodeOp::Unary { op: op_marker.0, src }
            } else if let Some(op_marker) = entity_ref.get::<crate::context::BinaryOpMarker>() {
                let bin_inputs = entity_ref
                    .get::<BinaryInputs>()
                    .expect("BinaryOp missing inputs");
                let lhs = entity_to_idx[&bin_inputs.left.entity()];
                let rhs = entity_to_idx[&bin_inputs.right.entity()];
                NodeOp::Binary { op: op_marker.0, lhs, rhs }
            } else {
                // Unknown node type — treat as constant
                let value = entity_ref.get::<Value>().map(|v| v.get()).unwrap_or(0.0);
                NodeOp::Constant(value)
            };

            nodes.push(node);
        }

        let output_node = entity_to_idx[&output.entity()];
        crate::compiled::CompiledGraph::new(nodes, output_node)
    }

    /// Clears all cached Taylor coefficients.
    ///
    /// Call this when you want to force recomputation of all derivatives,
    /// or to free memory after completing a computation.
    pub fn clear_cache(&mut self) {
        use crate::components::TaylorData;

        // Get all entities with TaylorData
        let entities: Vec<bevy_ecs::entity::Entity> = self
            .world
            .query::<bevy_ecs::entity::Entity>()
            .iter(&self.world)
            .collect();

        for entity in entities {
            if let Some(mut td) = self.world.get_mut::<TaylorData>(entity) {
                td.clear();
            }
        }
    }
}

impl Default for AutoDiff {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new_context() {
        let ad = AutoDiff::new();
        assert_eq!(ad.input_count(), 0);
    }

    #[test]
    fn test_create_var() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        assert_eq!(ad.eval(x), 5.0);
        assert!(ad.is_input(x));
        assert!(!ad.is_constant(x));
        assert_eq!(ad.input_count(), 1);
    }

    #[test]
    fn test_create_constant() {
        let mut ad = AutoDiff::new();
        let c = ad.constant(42.0);
        assert_eq!(ad.eval(c), 42.0);
        assert!(!ad.is_input(c));
        assert!(ad.is_constant(c));
        assert_eq!(ad.input_count(), 0); // Constants don't count as inputs
    }

    #[test]
    fn test_multiple_inputs() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);
        let z = ad.var(3.0);

        assert_eq!(ad.input_count(), 3);
        assert_eq!(ad.eval(x), 1.0);
        assert_eq!(ad.eval(y), 2.0);
        assert_eq!(ad.eval(z), 3.0);
    }

    #[test]
    fn test_set_input() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        assert_eq!(ad.eval(x), 5.0);

        ad.set_input(x, 10.0);
        assert_eq!(ad.eval(x), 10.0);
    }

    #[test]
    #[should_panic(expected = "set_input called on non-input variable")]
    fn test_set_input_on_constant_panics() {
        let mut ad = AutoDiff::new();
        let c = ad.constant(5.0);
        ad.set_input(c, 10.0);
    }

    // Binary operation tests
    #[test]
    fn test_add() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let z = ad.add(x, y);
        assert_eq!(ad.eval(z), 5.0);
    }

    #[test]
    fn test_sub() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let y = ad.var(3.0);
        let z = ad.sub(x, y);
        assert_eq!(ad.eval(z), 2.0);
    }

    #[test]
    fn test_mul() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);
        let y = ad.var(5.0);
        let z = ad.mul(x, y);
        assert_eq!(ad.eval(z), 20.0);
    }

    #[test]
    fn test_div() {
        let mut ad = AutoDiff::new();
        let x = ad.var(10.0);
        let y = ad.var(2.0);
        let z = ad.div(x, y);
        assert_eq!(ad.eval(z), 5.0);
    }

    #[test]
    fn test_pow() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let z = ad.pow(x, y);
        assert_eq!(ad.eval(z), 8.0);
    }

    // Unary operation tests
    #[test]
    fn test_neg() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let y = ad.neg(x);
        assert_eq!(ad.eval(y), -5.0);
    }

    #[test]
    fn test_sin_cos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let s = ad.sin(x);
        let c = ad.cos(x);

        assert_relative_eq!(ad.eval(s), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(c), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_ln() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let e = ad.exp(x);
        let l = ad.ln(x);

        assert_relative_eq!(ad.eval(e), std::f64::consts::E, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(l), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);
        let y = ad.sqrt(x);
        assert_eq!(ad.eval(y), 2.0);
    }

    #[test]
    fn test_sinh_cosh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let sh = ad.sinh(x);
        let ch = ad.cosh(x);

        assert_relative_eq!(ad.eval(sh), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(ch), 1.0, epsilon = 1e-10);
    }

    // Convenience operation tests
    #[test]
    fn test_square() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let y = ad.square(x);
        assert_eq!(ad.eval(y), 25.0);
    }

    #[test]
    fn test_powi() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.powi(x, 4);
        assert_eq!(ad.eval(y), 16.0);
    }

    #[test]
    fn test_powf() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);
        let y = ad.powf(x, 0.5);
        assert_eq!(ad.eval(y), 2.0);
    }

    // Complex expression tests
    #[test]
    fn test_complex_expression() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // f = (x + y) * (x - y) = x² - y²
        let sum = ad.add(x, y);
        let diff = ad.sub(x, y);
        let f = ad.mul(sum, diff);

        assert_eq!(ad.eval(f), -5.0); // 4 - 9 = -5
    }

    #[test]
    fn test_polynomial() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);

        // f = x³ + 2x² + 3x + 4
        let c4 = ad.constant(4.0);
        let c3 = ad.constant(3.0);
        let c2 = ad.constant(2.0);

        let x2 = ad.square(x);
        let x3 = ad.mul(x2, x);

        let term3 = x3;
        let term2 = ad.mul(c2, x2);
        let term1 = ad.mul(c3, x);
        let term0 = c4;

        let sum0 = ad.add(term1, term0);
        let sum1 = ad.add(term2, sum0);
        let f = ad.add(term3, sum1);

        // f(2) = 8 + 8 + 6 + 4 = 26
        assert_eq!(ad.eval(f), 26.0);
    }

    // Dependency tests
    #[test]
    fn test_dependencies_single_input() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.square(x);

        assert!(ad.depends_on(y, x));
        assert!(ad.depends_on(x, x)); // Self-dependency
    }

    #[test]
    fn test_dependencies_multiple_inputs() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);
        let z = ad.var(3.0);

        // f depends on x and y, but not z
        let f = ad.mul(x, y);

        assert!(ad.depends_on(f, x));
        assert!(ad.depends_on(f, y));
        assert!(!ad.depends_on(f, z));
    }

    #[test]
    fn test_dependencies_constant() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let c = ad.constant(5.0);

        // f = x * c
        let f = ad.mul(x, c);

        assert!(ad.depends_on(f, x));
        // Constant doesn't track as dependency in the bitmask sense
    }

    #[test]
    fn test_chain_rule_setup() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);

        // f = sin(x²)
        let x2 = ad.square(x);
        let f = ad.sin(x2);

        assert!(ad.depends_on(x2, x));
        assert!(ad.depends_on(f, x));
    }

    #[test]
    fn test_var_equality() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);

        assert_eq!(x, x);
        assert_ne!(x, y);
    }

    #[test]
    fn test_default_context() {
        let ad = AutoDiff::default();
        assert_eq!(ad.input_count(), 0);
    }

    // ===================
    // Derivative tests
    // ===================

    #[test]
    fn test_derivative_x_squared() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let y = ad.square(x); // y = x²

        // dy/dx = 2x = 6 at x=3
        assert_eq!(ad.derivative(y, x, 1), 6.0);

        // d²y/dx² = 2
        assert_eq!(ad.derivative(y, x, 2), 2.0);

        // d³y/dx³ = 0
        assert_relative_eq!(ad.derivative(y, x, 3), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_sin() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let y = ad.sin(x); // y = sin(x)

        // At x=0: sin(0)=0, cos(0)=1, -sin(0)=0, -cos(0)=-1
        assert_relative_eq!(ad.derivative(y, x, 0), 0.0, epsilon = 1e-10); // sin(0) = 0
        assert_relative_eq!(ad.derivative(y, x, 1), 1.0, epsilon = 1e-10); // cos(0) = 1
        assert_relative_eq!(ad.derivative(y, x, 2), 0.0, epsilon = 1e-10); // -sin(0) = 0
        assert_relative_eq!(ad.derivative(y, x, 3), -1.0, epsilon = 1e-10); // -cos(0) = -1
    }

    #[test]
    fn test_derivative_cos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let y = ad.cos(x); // y = cos(x)

        // At x=0: cos(0)=1, -sin(0)=0, -cos(0)=-1, sin(0)=0
        assert_relative_eq!(ad.derivative(y, x, 0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(ad.derivative(y, x, 1), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.derivative(y, x, 2), -1.0, epsilon = 1e-10);
        assert_relative_eq!(ad.derivative(y, x, 3), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_exp() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let y = ad.exp(x); // y = e^x

        // All derivatives of e^x at x=0 equal 1
        for k in 0..=5 {
            assert_relative_eq!(ad.derivative(y, x, k), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_derivative_ln() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.ln(x); // y = ln(x)

        // At x=1: ln(1)=0, 1/1=1, -1/1²=-1, 2!/1³=2, -3!/1⁴=-6
        assert_relative_eq!(ad.derivative(y, x, 0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.derivative(y, x, 1), 1.0, epsilon = 1e-10);
        assert_relative_eq!(ad.derivative(y, x, 2), -1.0, epsilon = 1e-10);
        assert_relative_eq!(ad.derivative(y, x, 3), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_sqrt() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);
        let y = ad.sqrt(x); // y = sqrt(x)

        // At x=4: sqrt(4)=2
        // d/dx sqrt(x) = 1/(2*sqrt(x)) = 1/4 = 0.25 at x=4
        assert_relative_eq!(ad.derivative(y, x, 0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(ad.derivative(y, x, 1), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_chain_rule() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let x2 = ad.square(x);
        let y = ad.sin(x2); // y = sin(x²)

        // dy/dx = cos(x²) * 2x = 0 at x=0
        assert_relative_eq!(ad.derivative(y, x, 1), 0.0, epsilon = 1e-10);

        // d²y/dx² = -sin(x²)*4x² + cos(x²)*2 = 2 at x=0
        assert_relative_eq!(ad.derivative(y, x, 2), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gradient() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);

        // f = x² + y²
        let x2 = ad.square(x);
        let y2 = ad.square(y);
        let f = ad.add(x2, y2);

        // ∇f = [2x, 2y] = [2, 4]
        let grad = ad.gradient(f);
        assert_eq!(grad.len(), 2);
        assert_relative_eq!(grad[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(grad[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_composition() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);

        // f = exp(sin(x))
        let sin_x = ad.sin(x);
        let f = ad.exp(sin_x);

        // f'(x) = exp(sin(x)) * cos(x)
        // At x=1: f'(1) = exp(sin(1)) * cos(1)
        let expected = (1.0_f64.sin()).exp() * 1.0_f64.cos();
        assert_relative_eq!(ad.derivative(f, x, 1), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_power() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.powi(x, 3); // y = x³

        // dy/dx = 3x² = 12 at x=2
        assert_relative_eq!(ad.derivative(y, x, 1), 12.0, epsilon = 1e-10);

        // d²y/dx² = 6x = 12 at x=2
        assert_relative_eq!(ad.derivative(y, x, 2), 12.0, epsilon = 1e-10);

        // d³y/dx³ = 6
        assert_relative_eq!(ad.derivative(y, x, 3), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_sqrt_as_power() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);
        let y = ad.powf(x, 0.5); // y = x^0.5 = sqrt(x)

        // Value: sqrt(4) = 2
        assert_relative_eq!(ad.eval(y), 2.0, epsilon = 1e-10);

        // dy/dx = 0.5 * x^(-0.5) = 0.5 / sqrt(4) = 0.25 at x=4
        assert_relative_eq!(ad.derivative(y, x, 1), 0.25, epsilon = 1e-10);
    }
}
