//! AutoDiff context - the main API for building computation graphs.

use std::collections::HashMap;

use bevy_ecs::world::World;
use bevy_entity_ptr::EntityHandle;

use crate::components::{
    BinaryInputs, BinaryOp, BinaryOpMarker, Dependencies, IsConstant, IsInput, UnaryInput,
    UnaryOp, UnaryOpMarker, Value, Variable,
};
use crate::graph::topology::topological_order;
use crate::var::Var;

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
/// let x = ad.var(2.0).unwrap();
/// let y = ad.var(3.0).unwrap();
///
/// // Build computation graph: f = x * y + x
/// let xy = ad.mul(x, y);
/// let f = ad.add(xy, x);
///
/// // Evaluate
/// assert_eq!(ad.eval(f).unwrap(), 8.0); // 2*3 + 2 = 8
/// ```
pub struct AutoDiff {
    /// The ECS world storing the computation graph.
    world: World,
    /// Counter for assigning input indices (for dependency tracking).
    input_count: usize,
    /// Input variables in creation order.
    inputs: Vec<Var>,
    /// Cached constant 0.0 entity (avoids creating duplicates in differentiate).
    cached_zero: Option<Var>,
    /// Cached constant 1.0 entity (avoids creating duplicates in differentiate).
    cached_one: Option<Var>,
}

impl AutoDiff {
    /// Creates a new, empty autodiff context.
    #[inline]
    pub fn new() -> Self {
        Self {
            world: World::new(),
            input_count: 0,
            inputs: Vec::new(),
            cached_zero: None,
            cached_one: None,
        }
    }

    /// Returns a reference to the underlying ECS world.
    /// Useful for advanced operations or debugging.
    #[inline]
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Returns a mutable reference to the underlying ECS world.
    #[inline]
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Creates a new input variable with the given initial value.
    ///
    /// Input variables are the leaves of the computation graph.
    /// Derivatives are computed with respect to these inputs.
    ///
    /// # Errors
    ///
    /// Returns [`InputLimitExceeded`](crate::error::AutoDiffError::InputLimitExceeded) if more than 64 input
    /// variables are created (the dependency bitmask is a u64).
    pub fn var(&mut self, value: f64) -> Result<Var, crate::error::AutoDiffError> {
        let input_index = self.input_count;
        if input_index >= 64 {
            return Err(crate::error::AutoDiffError::InputLimitExceeded { count: input_index });
        }
        self.input_count += 1;

        let entity = self
            .world
            .spawn((
                Variable,
                IsInput,
                Value::new(value),
                Dependencies::single(input_index),
            ))
            .id();

        let v = Var::new(entity);
        self.inputs.push(v);
        Ok(v)
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
            ))
            .id();

        Var::new(entity)
    }

    /// Evaluates a variable, returning its stored numerical value.
    ///
    /// For input variables, this returns the value set by [`var()`](Self::var) or
    /// [`set_input()`](Self::set_input). For computed variables, this returns the
    /// value computed at graph-construction time — it is **not** re-evaluated when
    /// inputs change. To re-evaluate at new input values, use
    /// [`CompiledGraph::eval()`](crate::compiled::CompiledGraph::eval).
    ///
    /// # Errors
    ///
    /// Returns [`MissingValue`](crate::error::AutoDiffError::MissingValue) if the variable does not have
    /// a `Value` component (e.g., the entity is from a different context).
    pub fn eval(&self, var: Var) -> Result<f64, crate::error::AutoDiffError> {
        self.world
            .entity(var.entity())
            .get::<Value>()
            .map(|v| v.get())
            .ok_or(crate::error::AutoDiffError::MissingValue)
    }

    /// Internal eval that panics on missing Value.
    /// Used by internal helpers (binary_op, unary_op, smart_*, is_const_value)
    /// where the caller guarantees the entity has a Value component.
    fn eval_unchecked(&self, var: Var) -> f64 {
        self.world
            .entity(var.entity())
            .get::<Value>()
            .map(|v| v.get())
            .expect("internal: Variable must have Value component")
    }

    /// Sets the value of an input variable.
    ///
    /// # Errors
    ///
    /// Returns [`NotAnInput`](crate::error::AutoDiffError::NotAnInput) if the variable is not an input variable.
    pub fn set_input(&mut self, var: Var, value: f64) -> Result<(), crate::error::AutoDiffError> {
        let entity = var.entity();

        // Verify it's an input
        if !self.world.entity(entity).contains::<IsInput>() {
            return Err(crate::error::AutoDiffError::NotAnInput);
        }

        // Update value
        if let Some(mut val) = self.world.entity_mut(entity).get_mut::<Value>() {
            val.set(value);
        }
        Ok(())
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

    /// Creates a new variable representing a^b using logarithmic differentiation.
    ///
    /// Primal evaluation is identical to [`pow`](Self::pow). The difference is in
    /// how symbolic derivatives are computed: this uses `d(a^b)/da = a^b · b · (da/a)`
    /// instead of `b · a^(b-1) · da`. The logarithmic form avoids catastrophic
    /// cancellation in f32 for second-order and higher derivatives, making it
    /// suitable for GPU evaluation via [`to_wgsl()`](crate::compiled::CompiledGraph::to_wgsl).
    ///
    /// **Note:** The intermediate `da/a` sub-expressions use standard division. This
    /// means f32 stability is guaranteed for second-order derivatives but may degrade
    /// at third order and above. For the common case (e.g., gravitational Hessians),
    /// second-order is the target and this is sufficient.
    ///
    /// **Requirement:** `base > 0`. Produces NaN if the base is zero or negative.
    pub fn pow_log(&mut self, base: Var, exponent: Var) -> Var {
        self.binary_op(BinaryOp::PowLog, base, exponent, |x, y| x.powf(y))
    }

    /// Creates a new variable representing a/b using logarithmic differentiation.
    ///
    /// Primal evaluation is identical to [`div`](Self::div). The difference is in
    /// how symbolic derivatives are computed: this uses `d(a/b) = (a/b) · (da/a - db/b)`
    /// instead of the quotient rule `(da·b - a·db) / b²`. The logarithmic form avoids
    /// catastrophic cancellation in f32 for second-order and higher derivatives.
    ///
    /// **Note:** The intermediate `da/a` and `db/b` sub-expressions use standard
    /// division. See [`pow_log`](Self::pow_log) for details on higher-order limits.
    ///
    /// **Requirement:** both `a` and `b` must be nonzero. Produces NaN otherwise.
    pub fn div_log(&mut self, a: Var, b: Var) -> Var {
        self.binary_op(BinaryOp::DivLog, a, b, |x, y| x / y)
    }

    /// Creates a new variable representing x^n (integer power) with logarithmic differentiation.
    ///
    /// See [`pow_log`](Self::pow_log) for details on when to use this.
    ///
    /// **Requirement:** `x > 0`. Produces NaN if x is zero or negative.
    pub fn powi_log(&mut self, x: Var, n: i32) -> Var {
        let n_const = self.constant(n as f64);
        self.pow_log(x, n_const)
    }

    /// Creates a new variable representing x^p (float power) with logarithmic differentiation.
    ///
    /// See [`pow_log`](Self::pow_log) for details on when to use this.
    ///
    /// **Requirement:** `x > 0`. Produces NaN if x is zero or negative.
    pub fn powf_log(&mut self, x: Var, p: f64) -> Var {
        let p_const = self.constant(p);
        self.pow_log(x, p_const)
    }

    /// Internal helper for creating binary operations.
    fn binary_op(&mut self, op: BinaryOp, a: Var, b: Var, f: fn(f64, f64) -> f64) -> Var {
        // Get values
        let a_val = self.eval_unchecked(a);
        let b_val = self.eval_unchecked(b);
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
        let x_val = self.eval_unchecked(x);
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

    /// Returns the input variables in creation order.
    #[inline]
    pub fn inputs(&self) -> &[Var] {
        &self.inputs
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
        output_deps
            .map(|d| (d.mask & input_mask) != 0)
            .unwrap_or(false)
    }

    /// Gathers current values of all input variables.
    fn gather_input_values(&self) -> Vec<f64> {
        self.inputs.iter().map(|&v| self.eval_unchecked(v)).collect()
    }

    /// Returns the position of `var` in `self.inputs`.
    ///
    /// # Panics
    /// Panics if `var` is not an input variable in this context.
    fn input_index(&self, var: Var) -> usize {
        self.inputs
            .iter()
            .position(|&v| v == var)
            .expect("Variable is not an input in this context")
    }

    /// Sets multiple input values at once.
    ///
    /// # Panics
    /// Panics if any variable is not an input variable.
    pub fn set_inputs(&mut self, inputs: &[(Var, f64)]) {
        for &(var, value) in inputs {
            self.set_input(var, value)
                .expect("set_inputs: variable is not an input");
        }
    }

    // =========================================================================
    // Symbolic Differentiation
    // =========================================================================

    /// Differentiates the computation graph symbolically.
    ///
    /// Creates NEW entities in the ECS world representing the derivative
    /// of `output` with respect to `wrt`. The returned `Var` points to
    /// the root of the derivative subgraph.
    ///
    /// For higher-order or mixed partial derivatives, call repeatedly:
    /// ```
    /// # use bevy_autodiff::AutoDiff;
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(2.0).unwrap();
    /// let y = ad.var(3.0).unwrap();
    /// let f = ad.mul(x, y); // f = x * y
    ///
    /// // First derivative: df/dx = y
    /// let dfdx = ad.differentiate(f, x).unwrap();
    /// assert_eq!(ad.eval(dfdx).unwrap(), 3.0);
    ///
    /// // Mixed partial: d²f/dxdy = 1
    /// let d2fdxdy = ad.differentiate(dfdx, y).unwrap();
    /// assert_eq!(ad.eval(d2fdxdy).unwrap(), 1.0);
    /// ```
    pub fn differentiate(&mut self, output: Var, wrt: Var) -> Result<Var, crate::error::AutoDiffError> {
        let order = topological_order(&self.world, output.entity())?;
        let mut derivs: HashMap<bevy_ecs::entity::Entity, Var> = HashMap::new();

        let zero = self.zero();
        let one = self.one();

        for &entity in &order {
            // Extract all info from the entity before any mutations
            let is_wrt = entity == wrt.entity();
            let is_input = self.world.entity(entity).contains::<IsInput>();
            let is_const = self.world.entity(entity).contains::<IsConstant>();
            let unary_op = self
                .world
                .entity(entity)
                .get::<UnaryOpMarker>()
                .map(|m| m.op());
            let binary_op = self
                .world
                .entity(entity)
                .get::<BinaryOpMarker>()
                .map(|m| m.op());
            let unary_input_entity = self
                .world
                .entity(entity)
                .get::<UnaryInput>()
                .map(|u| u.get().entity());
            let binary_input_entities = self
                .world
                .entity(entity)
                .get::<BinaryInputs>()
                .map(|b| (b.left.entity(), b.right.entity()));

            // Base cases
            if is_wrt {
                derivs.insert(entity, one);
                continue;
            }
            if is_input || is_const {
                derivs.insert(entity, zero);
                continue;
            }

            let z = Var::new(entity);

            if let Some(op) = unary_op {
                let a_entity = unary_input_entity
                    .expect("internal: unary op must have input entity");
                let a = Var::new(a_entity);
                let da = derivs[&a_entity];

                let dz = self.differentiate_unary(op, z, a, da, one);
                derivs.insert(entity, dz);
            } else if let Some(op) = binary_op {
                let (a_entity, b_entity) = binary_input_entities
                    .expect("internal: binary op must have input entities");
                let a = Var::new(a_entity);
                let b = Var::new(b_entity);
                let da = derivs[&a_entity];
                let db = derivs[&b_entity];

                let dz = self.differentiate_binary(op, z, a, b, da, db);
                derivs.insert(entity, dz);
            } else {
                derivs.insert(entity, zero);
            }
        }

        Ok(derivs[&output.entity()])
    }

    /// Apply chain rule for a unary operation: z = op(a), given da = d(a)/d(wrt).
    fn differentiate_unary(&mut self, op: UnaryOp, z: Var, a: Var, da: Var, one: Var) -> Var {
        match op {
            UnaryOp::Neg => {
                // d(-a) = -da
                self.smart_neg(da)
            }
            UnaryOp::Sin => {
                // d(sin(a)) = cos(a) * da
                let cos_a = self.cos(a);
                self.smart_mul(cos_a, da)
            }
            UnaryOp::Cos => {
                // d(cos(a)) = -sin(a) * da
                let sin_a = self.sin(a);
                let neg_sin_a = self.smart_neg(sin_a);
                self.smart_mul(neg_sin_a, da)
            }
            UnaryOp::Tan => {
                // d(tan(a)) = da / cos²(a)
                let cos_a = self.cos(a);
                let cos2_a = self.square(cos_a);
                self.smart_div(da, cos2_a)
            }
            UnaryOp::Exp => {
                // d(exp(a)) = exp(a) * da = z * da
                self.smart_mul(z, da)
            }
            UnaryOp::Ln => {
                // d(ln(a)) = da / a
                self.smart_div(da, a)
            }
            UnaryOp::Sqrt => {
                // d(sqrt(a)) = da / (2 * sqrt(a)) = da / (2 * z)
                let two = self.constant(2.0);
                let two_z = self.mul(two, z);
                self.smart_div(da, two_z)
            }
            UnaryOp::Sinh => {
                // d(sinh(a)) = cosh(a) * da
                let cosh_a = self.cosh(a);
                self.smart_mul(cosh_a, da)
            }
            UnaryOp::Cosh => {
                // d(cosh(a)) = sinh(a) * da
                let sinh_a = self.sinh(a);
                self.smart_mul(sinh_a, da)
            }
            UnaryOp::Tanh => {
                // d(tanh(a)) = (1 - tanh²(a)) * da = (1 - z²) * da
                let z2 = self.square(z);
                let one_minus_z2 = self.sub(one, z2);
                self.smart_mul(one_minus_z2, da)
            }
            UnaryOp::Asin => {
                // d(asin(a)) = da / sqrt(1 - a²)
                let a2 = self.square(a);
                let one_minus_a2 = self.sub(one, a2);
                let denom = self.sqrt(one_minus_a2);
                self.smart_div(da, denom)
            }
            UnaryOp::Acos => {
                // d(acos(a)) = -da / sqrt(1 - a²)
                let a2 = self.square(a);
                let one_minus_a2 = self.sub(one, a2);
                let denom = self.sqrt(one_minus_a2);
                let neg_da = self.smart_neg(da);
                self.smart_div(neg_da, denom)
            }
            UnaryOp::Atan => {
                // d(atan(a)) = da / (1 + a²)
                let a2 = self.square(a);
                let one_plus_a2 = self.add(one, a2);
                self.smart_div(da, one_plus_a2)
            }
            UnaryOp::Asinh => {
                // d(asinh(a)) = da / sqrt(a² + 1)
                let a2 = self.square(a);
                let a2_plus_1 = self.add(a2, one);
                let denom = self.sqrt(a2_plus_1);
                self.smart_div(da, denom)
            }
            UnaryOp::Acosh => {
                // d(acosh(a)) = da / sqrt(a² - 1)
                let a2 = self.square(a);
                let a2_minus_1 = self.sub(a2, one);
                let denom = self.sqrt(a2_minus_1);
                self.smart_div(da, denom)
            }
            UnaryOp::Atanh => {
                // d(atanh(a)) = da / (1 - a²)
                let a2 = self.square(a);
                let one_minus_a2 = self.sub(one, a2);
                self.smart_div(da, one_minus_a2)
            }
        }
    }

    /// Apply chain rule for a binary operation: z = op(a, b), given da, db.
    fn differentiate_binary(
        &mut self,
        op: BinaryOp,
        z: Var,
        a: Var,
        b: Var,
        da: Var,
        db: Var,
    ) -> Var {
        match op {
            BinaryOp::Add => {
                // d(a + b) = da + db
                self.smart_add(da, db)
            }
            BinaryOp::Sub => {
                // d(a - b) = da - db
                self.smart_sub(da, db)
            }
            BinaryOp::Mul => {
                // d(a * b) = da*b + a*db
                let term1 = self.smart_mul(da, b);
                let term2 = self.smart_mul(a, db);
                self.smart_add(term1, term2)
            }
            BinaryOp::Div => {
                // d(a / b) = (da*b - a*db) / b²
                let term1 = self.smart_mul(da, b);
                let term2 = self.smart_mul(a, db);
                let numer = self.smart_sub(term1, term2);
                let b2 = self.square(b);
                self.smart_div(numer, b2)
            }
            BinaryOp::Pow => {
                let da_is_zero = self.is_const_value(da, 0.0);
                let db_is_zero = self.is_const_value(db, 0.0);

                if da_is_zero && db_is_zero {
                    // Both inputs constant wrt wrt
                    self.constant(0.0)
                } else if db_is_zero {
                    // d(a^b) = b * a^(b-1) * da  (b constant wrt wrt)
                    // Special case: b == 0 means d(a^0)/d(wrt) = d(1)/d(wrt) = 0
                    if self.is_const_value(b, 0.0) {
                        return self.constant(0.0);
                    }
                    let one = self.one();
                    let b_minus_1 = self.sub(b, one);
                    let a_pow = self.pow(a, b_minus_1);
                    let b_times = self.smart_mul(b, a_pow);
                    self.smart_mul(b_times, da)
                } else if da_is_zero {
                    // d(a^b) = a^b * ln(a) * db  (a constant wrt wrt)
                    let ln_a = self.ln(a);
                    let z_ln_a = self.mul(z, ln_a);
                    self.smart_mul(z_ln_a, db)
                } else {
                    // d(a^b) = a^b * (db*ln(a) + b*da/a)  (general case)
                    let ln_a = self.ln(a);
                    let term1 = self.mul(db, ln_a);
                    let da_over_a = self.div(da, a);
                    let term2 = self.mul(b, da_over_a);
                    let inner = self.add(term1, term2);
                    self.mul(z, inner)
                }
            }
            BinaryOp::PowLog => {
                // Logarithmic differentiation: d(a^b) = a^b * (db*ln(a) + b*da/a)
                // The key difference from Pow: when b is constant, uses z*b*(da/a)
                // instead of b*a^(b-1)*da, avoiding catastrophic cancellation in f32.
                let da_is_zero = self.is_const_value(da, 0.0);
                let db_is_zero = self.is_const_value(db, 0.0);

                if da_is_zero && db_is_zero {
                    self.constant(0.0)
                } else if db_is_zero {
                    // d(a^b) = z * b * (da / a)  — logarithmic form
                    if self.is_const_value(b, 0.0) {
                        return self.constant(0.0);
                    }
                    let da_over_a = self.div(da, a);
                    let b_da_over_a = self.smart_mul(b, da_over_a);
                    self.smart_mul(z, b_da_over_a)
                } else if da_is_zero {
                    // d(a^b) = z * ln(a) * db  (same as Pow, already stable)
                    let ln_a = self.ln(a);
                    let z_ln_a = self.mul(z, ln_a);
                    self.smart_mul(z_ln_a, db)
                } else {
                    // d(a^b) = z * (db*ln(a) + b*da/a)  (same as Pow general)
                    let ln_a = self.ln(a);
                    let term1 = self.mul(db, ln_a);
                    let da_over_a = self.div(da, a);
                    let term2 = self.mul(b, da_over_a);
                    let inner = self.add(term1, term2);
                    self.mul(z, inner)
                }
            }
            BinaryOp::DivLog => {
                // Logarithmic differentiation: d(a/b) = (a/b) * (da/a - db/b)
                let da_is_zero = self.is_const_value(da, 0.0);
                let db_is_zero = self.is_const_value(db, 0.0);

                if da_is_zero && db_is_zero {
                    self.constant(0.0)
                } else if db_is_zero {
                    // d(a/b) = z * (da / a)  (b constant)
                    let da_over_a = self.div(da, a);
                    self.smart_mul(z, da_over_a)
                } else if da_is_zero {
                    // d(a/b) = z * (- db / b)  (a constant)
                    let db_over_b = self.div(db, b);
                    let neg_db_over_b = self.neg(db_over_b);
                    self.smart_mul(z, neg_db_over_b)
                } else {
                    // d(a/b) = z * (da/a - db/b)
                    let da_over_a = self.div(da, a);
                    let db_over_b = self.div(db, b);
                    let inner = self.sub(da_over_a, db_over_b);
                    self.mul(z, inner)
                }
            }
        }
    }

    // =========================================================================
    // Derivative API (built on differentiate)
    // =========================================================================

    /// Computes the n-th derivative of `output` with respect to `input`.
    ///
    /// Internally compiles to a [`CompiledGraph`](crate::compiled::CompiledGraph)
    /// and evaluates at the current input values, so results are correct
    /// even after calling [`set_input()`](Self::set_input).
    ///
    /// **Note:** Each call creates new derivative entities in the ECS world that
    /// are not cleaned up. For repeated evaluation at different points, use
    /// [`compile()`](Self::compile) or [`compile_order()`](Self::compile_order)
    /// to build the graph once and re-evaluate many times.
    ///
    /// ```
    /// # use bevy_autodiff::AutoDiff;
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(2.0).unwrap();
    /// let f = ad.powi(x, 3); // x³
    ///
    /// assert_eq!(ad.derivative(f, x, 1).unwrap(), 12.0); // 3x² = 12
    /// assert_eq!(ad.derivative(f, x, 2).unwrap(), 12.0); // 6x = 12
    /// assert_eq!(ad.derivative(f, x, 3).unwrap(), 6.0);  // 6
    /// ```
    pub fn derivative(&mut self, output: Var, input: Var, order: usize) -> Result<f64, crate::error::AutoDiffError> {
        if order == 0 {
            return self.eval(output);
        }
        let all_inputs: Vec<Var> = self.inputs.clone();
        let idx = self.input_index(input);
        let mut multi_index = vec![0usize; all_inputs.len()];
        multi_index[idx] = order;
        let mut cg = self.compile(output, &all_inputs, &[multi_index.clone()])?;
        let values = self.gather_input_values();
        cg.eval(&values)?;
        cg.partial(&multi_index)
    }

    /// Computes a mixed partial derivative specified by a multi-index.
    ///
    /// `multi_index[i]` is the number of times to differentiate with respect
    /// to `inputs[i]`. For example, `multi_index = [2, 1]` computes
    /// d³f / (dx² dy).
    ///
    /// Internally compiles to a [`CompiledGraph`](crate::compiled::CompiledGraph)
    /// and evaluates at the current input values, so results are correct
    /// even after calling [`set_input()`](Self::set_input).
    ///
    /// ```
    /// # use bevy_autodiff::AutoDiff;
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(1.0).unwrap();
    /// let y = ad.var(2.0).unwrap();
    /// let x2 = ad.square(x);
    /// let f = ad.mul(x2, y); // x² * y
    ///
    /// // d²f/dxdy = 2x = 2
    /// assert_eq!(ad.partial(f, &[1, 1], &[x, y]).unwrap(), 2.0);
    /// ```
    pub fn partial(&mut self, output: Var, multi_index: &[usize], inputs: &[Var]) -> Result<f64, crate::error::AutoDiffError> {
        assert_eq!(
            multi_index.len(),
            inputs.len(),
            "multi_index and inputs must have the same length"
        );
        let total_order: usize = multi_index.iter().sum();
        if total_order == 0 {
            return self.eval(output);
        }
        let all_inputs: Vec<Var> = self.inputs.clone();
        // Map user-provided multi_index to full multi-index over all inputs
        let mut full_mi = vec![0usize; all_inputs.len()];
        for (i, &count) in multi_index.iter().enumerate() {
            if count > 0 {
                let idx = self.input_index(inputs[i]);
                full_mi[idx] += count;
            }
        }
        let mut cg = self.compile(output, &all_inputs, &[full_mi.clone()])?;
        let values = self.gather_input_values();
        cg.eval(&values)?;
        cg.partial(&full_mi)
    }

    /// Computes the gradient of `output` with respect to all input variables.
    ///
    /// Returns a vector of partial derivatives in input creation order.
    ///
    /// Internally compiles to a [`CompiledGraph`](crate::compiled::CompiledGraph)
    /// and evaluates at the current input values, so results are correct
    /// even after calling [`set_input()`](Self::set_input).
    ///
    /// ```
    /// # use bevy_autodiff::AutoDiff;
    /// let mut ad = AutoDiff::new();
    /// let x = ad.var(1.0).unwrap();
    /// let y = ad.var(2.0).unwrap();
    /// let x2 = ad.square(x);
    /// let y2 = ad.square(y);
    /// let f = ad.add(x2, y2); // x² + y²
    ///
    /// let grad = ad.gradient(f).unwrap();
    /// assert_eq!(grad, vec![2.0, 4.0]); // [2x, 2y]
    /// ```
    pub fn gradient(&mut self, output: Var) -> Result<Vec<f64>, crate::error::AutoDiffError> {
        let all_inputs: Vec<Var> = self.inputs.clone();
        let n = all_inputs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let partials: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let mut mi = vec![0usize; n];
                mi[i] = 1;
                mi
            })
            .collect();
        let mut cg = self.compile(output, &all_inputs, &partials)?;
        let values = self.gather_input_values();
        cg.eval(&values)?;
        partials.iter().map(|mi| cg.partial(mi)).collect()
    }

    // =========================================================================
    // Compilation
    // =========================================================================

    /// Compiles the computation graph for fast repeated evaluation.
    ///
    /// Calls `differentiate()` to build derivative graphs for each requested
    /// partial, then flattens everything into a single forward-pass array.
    ///
    /// `inputs` specifies which variables are treated as inputs (their values
    /// change between evaluations). All other inputs are frozen at their
    /// current values.
    ///
    /// `partials` is a list of multi-indices specifying which partial
    /// derivatives to pre-compute. Each multi-index must have the same
    /// length as `inputs`.
    pub fn compile(
        &mut self,
        output: Var,
        inputs: &[Var],
        partials: &[Vec<usize>],
    ) -> Result<crate::compiled::CompiledGraph, crate::error::AutoDiffError> {
        use crate::compiled::{flatten_graph, CompiledGraph};
        use crate::graph::topology::topological_order_multi;

        // 1. Build derivative Vars for each requested partial
        let mut all_output_entities = vec![output.entity()];
        let mut partial_vars: Vec<(Vec<usize>, Var)> = Vec::new();

        for multi_index in partials {
            assert_eq!(
                multi_index.len(),
                inputs.len(),
                "multi_index length must match number of inputs"
            );
            let mut current = output;
            for (i, &count) in multi_index.iter().enumerate() {
                for _ in 0..count {
                    current = self.differentiate(current, inputs[i])?;
                }
            }
            partial_vars.push((multi_index.clone(), current));
            all_output_entities.push(current.entity());
        }

        // 2. Topological sort all entities reachable from any output
        let order = topological_order_multi(&self.world, &all_output_entities)?;

        // 3. Build input entity -> position mapping
        let input_to_pos: HashMap<bevy_ecs::entity::Entity, usize> = inputs
            .iter()
            .enumerate()
            .map(|(i, v)| (v.entity(), i))
            .collect();

        // 4. Flatten to Vec<NodeOp>
        let (nodes, entity_to_index) = flatten_graph(&self.world, &order, &input_to_pos);

        // 5. Store output indices
        let output_index = entity_to_index[&output.entity()];
        let partial_outputs: Vec<(Vec<usize>, usize)> = partial_vars
            .iter()
            .map(|(mi, var)| (mi.clone(), entity_to_index[&var.entity()]))
            .collect();

        Ok(CompiledGraph::new(nodes, inputs.len(), output_index, partial_outputs))
    }

    /// Compiles only the function value (no symbolic derivatives).
    ///
    /// Use `CompiledGraph::gradient()` for first-order partials via reverse mode,
    /// which computes the full gradient in a single backward pass regardless of
    /// the number of inputs.
    pub fn compile_primal(
        &mut self,
        output: Var,
        inputs: &[Var],
    ) -> Result<crate::compiled::CompiledGraph, crate::error::AutoDiffError> {
        self.compile(output, inputs, &[])
    }

    /// Compiles the computation graph with all partials up to `max_order`.
    ///
    /// Convenience wrapper around `compile()` that generates all
    /// multi-indices of total order 1 through `max_order`.
    pub fn compile_order(
        &mut self,
        output: Var,
        inputs: &[Var],
        max_order: usize,
    ) -> Result<crate::compiled::CompiledGraph, crate::error::AutoDiffError> {
        let partials = crate::compiled::generate_multi_indices(inputs.len(), max_order);
        self.compile(output, inputs, &partials)
    }

    // =========================================================================
    // Cached constants (reused across differentiate() calls)
    // =========================================================================

    /// Returns a cached constant 0.0 entity, creating it on first use.
    fn zero(&mut self) -> Var {
        if let Some(v) = self.cached_zero {
            return v;
        }
        let v = self.constant(0.0);
        self.cached_zero = Some(v);
        v
    }

    /// Returns a cached constant 1.0 entity, creating it on first use.
    fn one(&mut self) -> Var {
        if let Some(v) = self.cached_one {
            return v;
        }
        let v = self.constant(1.0);
        self.cached_one = Some(v);
        v
    }

    // =========================================================================
    // Smart Helpers (constant folding during differentiation)
    // =========================================================================

    /// Check if a Var is a constant with a specific value.
    ///
    /// Uses `==` (bitwise f64 equality), which means `is_const_value(v, 0.0)`
    /// returns `false` for NaN. This is intentional: the smart_* helpers use it
    /// to fold identities like `0 * x → 0`, which deliberately deviates from
    /// IEEE 754 (where `0 * NaN = NaN`). See smart_mul / smart_div doc comments.
    fn is_const_value(&self, v: Var, val: f64) -> bool {
        self.is_constant(v) && self.eval_unchecked(v) == val
    }

    /// Add with constant folding: skip if either is 0, fold if both constant.
    fn smart_add(&mut self, a: Var, b: Var) -> Var {
        if self.is_const_value(a, 0.0) {
            return b;
        }
        if self.is_const_value(b, 0.0) {
            return a;
        }
        if self.is_constant(a) && self.is_constant(b) {
            return self.constant(self.eval_unchecked(a) + self.eval_unchecked(b));
        }
        self.add(a, b)
    }

    /// Subtract with constant folding: skip if b is 0, negate if a is 0.
    fn smart_sub(&mut self, a: Var, b: Var) -> Var {
        if self.is_const_value(b, 0.0) {
            return a;
        }
        if self.is_const_value(a, 0.0) {
            return self.smart_neg(b);
        }
        if self.is_constant(a) && self.is_constant(b) {
            return self.constant(self.eval_unchecked(a) - self.eval_unchecked(b));
        }
        self.sub(a, b)
    }

    /// Multiply with constant folding: return 0 if either is 0, identity if 1.
    ///
    /// Note: This deliberately deviates from IEEE 754 semantics where 0.0 * NaN = NaN
    /// and 0.0 * Inf = NaN. In symbolic differentiation, folding 0 * (anything) = 0
    /// is correct because a zero derivative means the branch does not contribute,
    /// regardless of the other factor's value at the evaluation point.
    fn smart_mul(&mut self, a: Var, b: Var) -> Var {
        if self.is_const_value(a, 0.0) || self.is_const_value(b, 0.0) {
            return self.constant(0.0);
        }
        if self.is_const_value(a, 1.0) {
            return b;
        }
        if self.is_const_value(b, 1.0) {
            return a;
        }
        if self.is_constant(a) && self.is_constant(b) {
            return self.constant(self.eval_unchecked(a) * self.eval_unchecked(b));
        }
        self.mul(a, b)
    }

    /// Negate with constant folding: skip if 0, fold if constant.
    fn smart_neg(&mut self, a: Var) -> Var {
        if self.is_const_value(a, 0.0) {
            return a;
        }
        if self.is_constant(a) {
            return self.constant(-self.eval_unchecked(a));
        }
        self.neg(a)
    }

    /// Divide with constant folding: return 0 if numerator is 0, identity if denom is 1.
    ///
    /// Note: Like smart_mul, this deliberately deviates from IEEE 754 where 0.0/0.0 = NaN.
    /// In symbolic differentiation, a zero numerator means the derivative term is zero
    /// regardless of the denominator. This prevents NaN poisoning the derivative graph
    /// when subexpressions hit domain boundaries.
    fn smart_div(&mut self, a: Var, b: Var) -> Var {
        if self.is_const_value(a, 0.0) {
            return self.constant(0.0);
        }
        if self.is_const_value(b, 1.0) {
            return a;
        }
        if self.is_constant(a) && self.is_constant(b) {
            return self.constant(self.eval_unchecked(a) / self.eval_unchecked(b));
        }
        self.div(a, b)
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
        let x = ad.var(5.0).unwrap();
        assert_eq!(ad.eval(x).unwrap(), 5.0);
        assert!(ad.is_input(x));
        assert!(!ad.is_constant(x));
        assert_eq!(ad.input_count(), 1);
    }

    #[test]
    fn test_create_constant() {
        let mut ad = AutoDiff::new();
        let c = ad.constant(42.0);
        assert_eq!(ad.eval(c).unwrap(), 42.0);
        assert!(!ad.is_input(c));
        assert!(ad.is_constant(c));
        assert_eq!(ad.input_count(), 0); // Constants don't count as inputs
    }

    #[test]
    fn test_multiple_inputs() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let y = ad.var(2.0).unwrap();
        let z = ad.var(3.0).unwrap();

        assert_eq!(ad.input_count(), 3);
        assert_eq!(ad.eval(x).unwrap(), 1.0);
        assert_eq!(ad.eval(y).unwrap(), 2.0);
        assert_eq!(ad.eval(z).unwrap(), 3.0);
    }

    #[test]
    fn test_inputs_tracking() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let y = ad.var(2.0).unwrap();
        assert_eq!(ad.inputs(), &[x, y]);
    }

    #[test]
    fn test_set_input() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();
        assert_eq!(ad.eval(x).unwrap(), 5.0);

        ad.set_input(x, 10.0).unwrap();
        assert_eq!(ad.eval(x).unwrap(), 10.0);
    }

    #[test]
    #[should_panic(expected = "NotAnInput")]
    fn test_set_input_on_constant_panics() {
        let mut ad = AutoDiff::new();
        let c = ad.constant(5.0);
        ad.set_input(c, 10.0).unwrap();
    }

    // Binary operation tests
    #[test]
    fn test_add() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let z = ad.add(x, y);
        assert_eq!(ad.eval(z).unwrap(), 5.0);
    }

    #[test]
    fn test_sub() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let z = ad.sub(x, y);
        assert_eq!(ad.eval(z).unwrap(), 2.0);
    }

    #[test]
    fn test_mul() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0).unwrap();
        let y = ad.var(5.0).unwrap();
        let z = ad.mul(x, y);
        assert_eq!(ad.eval(z).unwrap(), 20.0);
    }

    #[test]
    fn test_div() {
        let mut ad = AutoDiff::new();
        let x = ad.var(10.0).unwrap();
        let y = ad.var(2.0).unwrap();
        let z = ad.div(x, y);
        assert_eq!(ad.eval(z).unwrap(), 5.0);
    }

    #[test]
    fn test_pow() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let z = ad.pow(x, y);
        assert_eq!(ad.eval(z).unwrap(), 8.0);
    }

    // Unary operation tests
    #[test]
    fn test_neg() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();
        let y = ad.neg(x);
        assert_eq!(ad.eval(y).unwrap(), -5.0);
    }

    #[test]
    fn test_sin_cos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let s = ad.sin(x);
        let c = ad.cos(x);

        assert_relative_eq!(ad.eval(s).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(c).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_ln() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let e = ad.exp(x);
        let l = ad.ln(x);

        assert_relative_eq!(ad.eval(e).unwrap(), std::f64::consts::E, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(l).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0).unwrap();
        let y = ad.sqrt(x);
        assert_eq!(ad.eval(y).unwrap(), 2.0);
    }

    #[test]
    fn test_sinh_cosh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let sh = ad.sinh(x);
        let ch = ad.cosh(x);

        assert_relative_eq!(ad.eval(sh).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(ch).unwrap(), 1.0, epsilon = 1e-10);
    }

    // Convenience operation tests
    #[test]
    fn test_square() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();
        let y = ad.square(x);
        assert_eq!(ad.eval(y).unwrap(), 25.0);
    }

    #[test]
    fn test_powi() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.powi(x, 4);
        assert_eq!(ad.eval(y).unwrap(), 16.0);
    }

    #[test]
    fn test_powf() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0).unwrap();
        let y = ad.powf(x, 0.5);
        assert_eq!(ad.eval(y).unwrap(), 2.0);
    }

    // Complex expression tests
    #[test]
    fn test_complex_expression() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        // f = (x + y) * (x - y) = x² - y²
        let sum = ad.add(x, y);
        let diff = ad.sub(x, y);
        let f = ad.mul(sum, diff);

        assert_eq!(ad.eval(f).unwrap(), -5.0); // 4 - 9 = -5
    }

    #[test]
    fn test_polynomial() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();

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
        assert_eq!(ad.eval(f).unwrap(), 26.0);
    }

    // Dependency tests
    #[test]
    fn test_dependencies_single_input() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let y = ad.square(x);

        assert!(ad.depends_on(y, x));
        assert!(ad.depends_on(x, x)); // Self-dependency
    }

    #[test]
    fn test_dependencies_multiple_inputs() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let y = ad.var(2.0).unwrap();
        let z = ad.var(3.0).unwrap();

        // f depends on x and y, but not z
        let f = ad.mul(x, y);

        assert!(ad.depends_on(f, x));
        assert!(ad.depends_on(f, y));
        assert!(!ad.depends_on(f, z));
    }

    #[test]
    fn test_dependencies_constant() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let c = ad.constant(5.0);

        // f = x * c
        let f = ad.mul(x, c);

        assert!(ad.depends_on(f, x));
        // Constant doesn't track as dependency in the bitmask sense
    }

    #[test]
    fn test_chain_rule_setup() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();

        // f = sin(x²)
        let x2 = ad.square(x);
        let f = ad.sin(x2);

        assert!(ad.depends_on(x2, x));
        assert!(ad.depends_on(f, x));
    }

    #[test]
    fn test_var_equality() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let y = ad.var(2.0).unwrap();

        assert_eq!(x, x);
        assert_ne!(x, y);
    }

    #[test]
    fn test_default_context() {
        let ad = AutoDiff::default();
        assert_eq!(ad.input_count(), 0);
    }

    // =========================================================================
    // Differentiation tests
    // =========================================================================

    #[test]
    fn test_diff_identity() {
        // d/dx(x) = 1
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let dxdx = ad.differentiate(x, x).unwrap();
        assert_eq!(ad.eval(dxdx).unwrap(), 1.0);
    }

    #[test]
    fn test_diff_constant() {
        // d/dx(c) = 0
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let c = ad.constant(5.0);
        let dc = ad.differentiate(c, x).unwrap();
        assert_eq!(ad.eval(dc).unwrap(), 0.0);
    }

    #[test]
    fn test_diff_other_input() {
        // d/dx(y) = 0
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let y = ad.var(5.0).unwrap();
        let dy = ad.differentiate(y, x).unwrap();
        assert_eq!(ad.eval(dy).unwrap(), 0.0);
    }

    #[test]
    fn test_diff_neg() {
        // d/dx(-x) = -1
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let f = ad.neg(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_eq!(ad.eval(df).unwrap(), -1.0);
    }

    #[test]
    fn test_diff_add() {
        // d/dx(x + c) = 1
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let c = ad.constant(5.0);
        let f = ad.add(x, c);
        let df = ad.differentiate(f, x).unwrap();
        assert_eq!(ad.eval(df).unwrap(), 1.0);
    }

    #[test]
    fn test_diff_sub() {
        // d/dx(x - c) = 1
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let c = ad.constant(5.0);
        let f = ad.sub(x, c);
        let df = ad.differentiate(f, x).unwrap();
        assert_eq!(ad.eval(df).unwrap(), 1.0);
    }

    #[test]
    fn test_diff_mul_by_constant() {
        // d/dx(c * x) = c
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let c = ad.constant(5.0);
        let f = ad.mul(c, x);
        let df = ad.differentiate(f, x).unwrap();
        assert_eq!(ad.eval(df).unwrap(), 5.0);
    }

    #[test]
    fn test_diff_mul_two_vars() {
        // d/dx(x * y) = y
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.mul(x, y);
        let df = ad.differentiate(f, x).unwrap();
        assert_eq!(ad.eval(df).unwrap(), 3.0);
    }

    #[test]
    fn test_diff_div_by_constant() {
        // d/dx(x / c) = 1/c
        let mut ad = AutoDiff::new();
        let x = ad.var(6.0).unwrap();
        let c = ad.constant(3.0);
        let f = ad.div(x, c);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_square() {
        // d/dx(x²) = 2x at x=3 → 6
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let f = ad.square(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_eq!(ad.eval(df).unwrap(), 6.0);
    }

    #[test]
    fn test_diff_pow_const_exp() {
        // d/dx(x^3) = 3x² at x=2 → 12
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let f = ad.powi(x, 3);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_sin() {
        // d/dx(sin(x)) = cos(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = ad.sin(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 0.5_f64.cos(), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_cos() {
        // d/dx(cos(x)) = -sin(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = ad.cos(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), -(0.5_f64.sin()), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_tan() {
        // d/dx(tan(x)) = 1/cos²(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = ad.tan(x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = 1.0 / (0.5_f64.cos().powi(2));
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_exp() {
        // d/dx(exp(x)) = exp(x) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = ad.exp(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 1.0_f64.exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_ln() {
        // d/dx(ln(x)) = 1/x at x=2
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let f = ad.ln(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_sqrt() {
        // d/dx(sqrt(x)) = 1/(2*sqrt(x)) at x=4 → 0.25
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0).unwrap();
        let f = ad.sqrt(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_sinh() {
        // d/dx(sinh(x)) = cosh(x) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = ad.sinh(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 1.0_f64.cosh(), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_cosh() {
        // d/dx(cosh(x)) = sinh(x) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = ad.cosh(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 1.0_f64.sinh(), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_tanh() {
        // d/dx(tanh(x)) = 1 - tanh²(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = ad.tanh(x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = 1.0 - 0.5_f64.tanh().powi(2);
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_asin() {
        // d/dx(asin(x)) = 1/sqrt(1-x²) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = ad.asin(x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = 1.0 / (1.0 - 0.25_f64).sqrt();
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_acos() {
        // d/dx(acos(x)) = -1/sqrt(1-x²) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = ad.acos(x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = -1.0 / (1.0 - 0.25_f64).sqrt();
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_atan() {
        // d/dx(atan(x)) = 1/(1+x²) at x=1 → 0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = ad.atan(x);
        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_asinh() {
        // d/dx(asinh(x)) = 1/sqrt(x²+1) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = ad.asinh(x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = 1.0 / (2.0_f64).sqrt();
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_acosh() {
        // d/dx(acosh(x)) = 1/sqrt(x²-1) at x=2
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let f = ad.acosh(x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = 1.0 / (3.0_f64).sqrt();
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_atanh() {
        // d/dx(atanh(x)) = 1/(1-x²) at x=0.5 → 1/0.75
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = ad.atanh(x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = 1.0 / (1.0 - 0.25);
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_chain_rule() {
        // d/dx(sin(x²)) = 2x * cos(x²) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let x2 = ad.square(x);
        let f = ad.sin(x2);
        let df = ad.differentiate(f, x).unwrap();
        let expected = 2.0 * 1.0_f64.cos(); // 2*1 * cos(1)
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_product_rule() {
        // d/dx(x * sin(x)) = sin(x) + x * cos(x) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let sin_x = ad.sin(x);
        let f = ad.mul(x, sin_x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = 1.0_f64.sin() + 1.0 * 1.0_f64.cos();
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_quotient_rule() {
        // d/dx(sin(x)/x) = (cos(x)*x - sin(x)) / x² at x=2
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let sin_x = ad.sin(x);
        let f = ad.div(sin_x, x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = (2.0_f64.cos() * 2.0 - 2.0_f64.sin()) / 4.0;
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_second_derivative() {
        // d²/dx²(x³) = 6x at x=2 → 12
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let f = ad.powi(x, 3);
        let df = ad.differentiate(f, x).unwrap();
        let d2f = ad.differentiate(df, x).unwrap();
        assert_relative_eq!(ad.eval(d2f).unwrap(), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_third_derivative() {
        // d³/dx³(x⁴) = 24x at x=1 → 24
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = ad.powi(x, 4);
        let df = ad.differentiate(f, x).unwrap();
        let d2f = ad.differentiate(df, x).unwrap();
        let d3f = ad.differentiate(d2f, x).unwrap();
        assert_relative_eq!(ad.eval(d3f).unwrap(), 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_mixed_partial() {
        // d²/dxdy(x * y) = 1
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.mul(x, y);
        let dfdx = ad.differentiate(f, x).unwrap();
        let d2fdxdy = ad.differentiate(dfdx, y).unwrap();
        assert_relative_eq!(ad.eval(d2fdxdy).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_mixed_partial_symmetry() {
        // d²f/dxdy = d²f/dydx for f = x² * y + y³
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let y = ad.var(2.0).unwrap();

        let x2 = ad.square(x);
        let x2y = ad.mul(x2, y);
        let y2 = ad.square(y);
        let y3 = ad.mul(y, y2);
        let f = ad.add(x2y, y3);

        let dfdx = ad.differentiate(f, x).unwrap();
        let d2fdxdy = ad.differentiate(dfdx, y).unwrap();

        let dfdy = ad.differentiate(f, y).unwrap();
        let d2fdydx = ad.differentiate(dfdy, x).unwrap();

        assert_relative_eq!(ad.eval(d2fdxdy).unwrap(), ad.eval(d2fdydx).unwrap(), epsilon = 1e-10);
        // d²f/dxdy = 2x at (1,2) → 2
        assert_relative_eq!(ad.eval(d2fdxdy).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_exp_chain() {
        // d/dx(exp(sin(x))) = exp(sin(x)) * cos(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let sin_x = ad.sin(x);
        let f = ad.exp(sin_x);
        let df = ad.differentiate(f, x).unwrap();
        let expected = (0.5_f64.sin()).exp() * 0.5_f64.cos();
        assert_relative_eq!(ad.eval(df).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_polynomial() {
        // f = x³ + 2x² + 3x + 4
        // f'= 3x² + 4x + 3
        // At x=2: f' = 12 + 8 + 3 = 23
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let c4 = ad.constant(4.0);
        let c3 = ad.constant(3.0);
        let c2 = ad.constant(2.0);

        let x2 = ad.square(x);
        let x3 = ad.mul(x2, x);
        let term2 = ad.mul(c2, x2);
        let term1 = ad.mul(c3, x);

        let sum0 = ad.add(term1, c4);
        let sum1 = ad.add(term2, sum0);
        let f = ad.add(x3, sum1);

        let df = ad.differentiate(f, x).unwrap();
        assert_relative_eq!(ad.eval(df).unwrap(), 23.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_rosenbrock() {
        // f = (1-x)² + 100(y-x²)²
        // df/dx = -2(1-x) + 100 * 2(y-x²) * (-2x) = 2(x-1) - 400x(y-x²)
        // At (1,1): df/dx = 0
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let y = ad.var(1.0).unwrap();
        let one = ad.constant(1.0);
        let hundred = ad.constant(100.0);

        let one_minus_x = ad.sub(one, x);
        let term1 = ad.square(one_minus_x);
        let x_sq = ad.square(x);
        let y_minus_x_sq = ad.sub(y, x_sq);
        let term2_inner = ad.square(y_minus_x_sq);
        let term2 = ad.mul(hundred, term2_inner);
        let f = ad.add(term1, term2);

        let dfdx = ad.differentiate(f, x).unwrap();
        let dfdy = ad.differentiate(f, y).unwrap();

        assert_relative_eq!(ad.eval(dfdx).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(dfdy).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_rosenbrock_away_from_min() {
        // f = (1-x)² + 100(y-x²)²
        // df/dx = 2(x-1) - 400x(y-x²)
        // df/dy = 200(y-x²)
        // At (0,0): df/dx = -2, df/dy = 0
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let y = ad.var(0.0).unwrap();
        let one = ad.constant(1.0);
        let hundred = ad.constant(100.0);

        let one_minus_x = ad.sub(one, x);
        let term1 = ad.square(one_minus_x);
        let x_sq = ad.square(x);
        let y_minus_x_sq = ad.sub(y, x_sq);
        let term2_inner = ad.square(y_minus_x_sq);
        let term2 = ad.mul(hundred, term2_inner);
        let f = ad.add(term1, term2);

        let dfdx = ad.differentiate(f, x).unwrap();
        let dfdy = ad.differentiate(f, y).unwrap();

        assert_relative_eq!(ad.eval(dfdx).unwrap(), -2.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(dfdy).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_constant_folding() {
        // Verify that differentiating a constant wrt x produces a
        // constant entity, not just a value of 0
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let c = ad.constant(5.0);
        let dc = ad.differentiate(c, x).unwrap();
        assert!(ad.is_constant(dc));
        assert_eq!(ad.eval(dc).unwrap(), 0.0);
    }

    // =========================================================================
    // CompiledGraph tests
    // =========================================================================

    #[test]
    fn test_compile_value_only() {
        // f = x * y, no partials
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.mul(x, y);

        let mut cg = ad.compile(f, &[x, y], &[]).unwrap();
        cg.eval(&[2.0, 3.0]).unwrap();
        assert_eq!(cg.value(), 6.0);
        cg.eval(&[4.0, 5.0]).unwrap();
        assert_eq!(cg.value(), 20.0);
    }

    #[test]
    fn test_compile_with_derivatives() {
        // f = x², df/dx = 2x, d²f/dx² = 2
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let f = ad.square(x);

        let mut cg = ad.compile(f, &[x], &[vec![1], vec![2]]).unwrap();
        cg.eval(&[3.0]).unwrap();
        assert_eq!(cg.value(), 9.0);
        assert_eq!(cg.partial(&[1]).unwrap(), 6.0);
        assert_eq!(cg.partial(&[2]).unwrap(), 2.0);

        // Different input
        cg.eval(&[5.0]).unwrap();
        assert_eq!(cg.value(), 25.0);
        assert_relative_eq!(cg.partial(&[1]).unwrap(), 10.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[2]).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compile_order_two_vars() {
        // f = x * y, compile all partials up to order 2
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.mul(x, y);

        let mut cg = ad.compile_order(f, &[x, y], 2).unwrap();
        cg.eval(&[2.0, 3.0]).unwrap();

        assert_eq!(cg.value(), 6.0);
        assert_relative_eq!(cg.partial(&[1, 0]).unwrap(), 3.0, epsilon = 1e-10); // df/dx = y
        assert_relative_eq!(cg.partial(&[0, 1]).unwrap(), 2.0, epsilon = 1e-10); // df/dy = x
        assert_relative_eq!(cg.partial(&[1, 1]).unwrap(), 1.0, epsilon = 1e-10); // d²f/dxdy = 1
        assert_relative_eq!(cg.partial(&[2, 0]).unwrap(), 0.0, epsilon = 1e-10); // d²f/dx² = 0
        assert_relative_eq!(cg.partial(&[0, 2]).unwrap(), 0.0, epsilon = 1e-10); // d²f/dy² = 0
    }

    #[test]
    fn test_compile_sin_exp() {
        // f = sin(exp(x))
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let exp_x = ad.exp(x);
        let f = ad.sin(exp_x);

        let mut cg = ad.compile(f, &[x], &[vec![1], vec![2]]).unwrap();

        // Eval at x=0
        cg.eval(&[0.0]).unwrap();
        assert_relative_eq!(cg.value(), 1.0_f64.sin(), epsilon = 1e-10);
        // f'(x) = cos(exp(x)) * exp(x)
        let expected_d1 = 1.0_f64.cos() * 1.0_f64;
        assert_relative_eq!(cg.partial(&[1]).unwrap(), expected_d1, epsilon = 1e-10);

        // Eval at different point
        cg.eval(&[0.5]).unwrap();
        let e05 = 0.5_f64.exp();
        assert_relative_eq!(cg.value(), e05.sin(), epsilon = 1e-10);
        let expected_d1 = e05.cos() * e05;
        assert_relative_eq!(cg.partial(&[1]).unwrap(), expected_d1, epsilon = 1e-10);
    }

    #[test]
    fn test_compile_matches_ecs() {
        // Verify compiled output matches ECS evaluation for Rosenbrock
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let y = ad.var(0.8).unwrap();
        let one = ad.constant(1.0);
        let hundred = ad.constant(100.0);

        let one_minus_x = ad.sub(one, x);
        let term1 = ad.square(one_minus_x);
        let x_sq = ad.square(x);
        let y_minus_x_sq = ad.sub(y, x_sq);
        let term2_inner = ad.square(y_minus_x_sq);
        let term2 = ad.mul(hundred, term2_inner);
        let f = ad.add(term1, term2);

        // ECS derivatives
        let ecs_val = ad.eval(f).unwrap();
        let ecs_dfdx = ad.derivative(f, x, 1).unwrap();
        let ecs_dfdy = ad.derivative(f, y, 1).unwrap();

        // Compiled derivatives
        let mut cg = ad.compile_order(f, &[x, y], 1).unwrap();
        cg.eval(&[0.5, 0.8]).unwrap();

        assert_relative_eq!(cg.value(), ecs_val, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1, 0]).unwrap(), ecs_dfdx, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[0, 1]).unwrap(), ecs_dfdy, epsilon = 1e-10);
    }

    #[test]
    fn test_pow_variable_exponent_compiled() {
        // f(x,y) = x^y, df/dx = y * x^(y-1), df/dy = x^y * ln(x)
        // Verify that the derivative graph is symbolic in y (not frozen at compile-time y).
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.pow(x, y);

        let mut cg = ad.compile_order(f, &[x, y], 1).unwrap();

        // At (x=2, y=3): f = 8, df/dx = 3*4 = 12, df/dy = 8*ln(2)
        cg.eval(&[2.0, 3.0]).unwrap();
        assert_relative_eq!(cg.value(), 8.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1, 0]).unwrap(), 12.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[0, 1]).unwrap(), 8.0 * 2.0_f64.ln(), epsilon = 1e-10);

        // Re-evaluate at (x=3, y=2): f = 9, df/dx = 2*3 = 6, df/dy = 9*ln(3)
        cg.eval(&[3.0, 2.0]).unwrap();
        assert_relative_eq!(cg.value(), 9.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[1, 0]).unwrap(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(cg.partial(&[0, 1]).unwrap(), 9.0 * 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_smart_mul_nan_folding() {
        // smart_mul (used during differentiation) folds 0 * x → 0, even if
        // x could be NaN. This is a deliberate deviation from IEEE 754 (where
        // 0 * NaN = NaN) because in symbolic differentiation, zero derivative
        // terms are structurally zero.
        //
        // Exercise this via differentiate: d/dy(x) = 0 (constant-folded).
        // The chain rule produces 0 * (something), and smart_mul folds it.
        let mut ad = AutoDiff::new();
        let x = ad.var(f64::NAN).unwrap();
        let y = ad.var(1.0).unwrap();

        // f = x + y; df/dx at x=NaN should still be the constant 1
        // (smart_mul folds 0*... terms, smart_add collapses 0+1 → 1)
        let f = ad.add(x, y);
        let df_dy = ad.differentiate(f, y).unwrap();
        assert!(
            ad.is_constant(df_dy),
            "d(x+y)/dy should be constant-folded to 1"
        );
        assert_eq!(ad.eval(df_dy).unwrap(), 1.0);
    }

    // =========================================================================
    // Reverse-mode cross-validation tests
    // =========================================================================

    /// Helper: builds a compiled graph for f(x,y) and checks that
    /// reverse-mode gradient matches forward-mode symbolic partials.
    fn assert_reverse_matches_forward(
        build_fn: impl FnOnce(&mut AutoDiff, Var, Var) -> Var,
        points: &[(f64, f64)],
        epsilon: f64,
    ) {
        let mut ad = AutoDiff::new();
        let x = ad.var(points[0].0).unwrap();
        let y = ad.var(points[0].1).unwrap();
        let f = build_fn(&mut ad, x, y);

        // Forward-mode: compile with order-1 symbolic partials
        let mut cg_fwd = ad.compile_order(f, &[x, y], 1).unwrap();

        // Reverse-mode: compile primal only
        let mut cg_rev = ad.compile_primal(f, &[x, y]).unwrap();

        for &(xv, yv) in points {
            cg_fwd.eval(&[xv, yv]).unwrap();
            let fwd_dfdx = cg_fwd.partial(&[1, 0]).unwrap();
            let fwd_dfdy = cg_fwd.partial(&[0, 1]).unwrap();

            cg_rev.eval(&[xv, yv]).unwrap();
            let rev_grad = cg_rev.gradient();

            assert_relative_eq!(rev_grad[0], fwd_dfdx, epsilon = epsilon);
            assert_relative_eq!(rev_grad[1], fwd_dfdy, epsilon = epsilon);
        }
    }

    #[test]
    fn test_reverse_vs_forward_mul() {
        // f = x * y
        assert_reverse_matches_forward(
            |ad, x, y| ad.mul(x, y),
            &[(2.0, 3.0), (0.5, -1.5), (1.0, 1.0)],
            1e-10,
        );
    }

    #[test]
    fn test_reverse_vs_forward_add() {
        // f = x + y
        assert_reverse_matches_forward(
            |ad, x, y| ad.add(x, y),
            &[(2.0, 3.0), (-1.0, 5.0)],
            1e-10,
        );
    }

    #[test]
    fn test_reverse_vs_forward_sub() {
        // f = x - y
        assert_reverse_matches_forward(
            |ad, x, y| ad.sub(x, y),
            &[(2.0, 3.0), (5.0, 1.0)],
            1e-10,
        );
    }

    #[test]
    fn test_reverse_vs_forward_div() {
        // f = x / y
        assert_reverse_matches_forward(
            |ad, x, y| ad.div(x, y),
            &[(6.0, 3.0), (1.0, 2.0)],
            1e-10,
        );
    }

    #[test]
    fn test_reverse_vs_forward_pow() {
        // f = x^y
        assert_reverse_matches_forward(
            |ad, x, y| ad.pow(x, y),
            &[(2.0, 3.0), (3.0, 2.0)],
            1e-10,
        );
    }

    #[test]
    fn test_reverse_vs_forward_rosenbrock() {
        // f = (1-x)² + 100(y-x²)²
        assert_reverse_matches_forward(
            |ad, x, y| {
                let one = ad.constant(1.0);
                let hundred = ad.constant(100.0);
                let one_minus_x = ad.sub(one, x);
                let term1 = ad.square(one_minus_x);
                let x_sq = ad.square(x);
                let y_minus_x_sq = ad.sub(y, x_sq);
                let term2_inner = ad.square(y_minus_x_sq);
                let term2 = ad.mul(hundred, term2_inner);
                ad.add(term1, term2)
            },
            &[(0.0, 0.0), (1.0, 1.0), (0.5, 0.8), (-1.0, 2.0)],
            1e-10,
        );
    }

    #[test]
    fn test_reverse_vs_forward_sin_cos_composition() {
        // f = sin(x) * cos(y)
        assert_reverse_matches_forward(
            |ad, x, y| {
                let sx = ad.sin(x);
                let cy = ad.cos(y);
                ad.mul(sx, cy)
            },
            &[(0.5, 0.7), (1.0, 2.0), (0.0, 0.0)],
            1e-10,
        );
    }

    #[test]
    fn test_reverse_vs_forward_exp_chain() {
        // f = exp(x * y)
        assert_reverse_matches_forward(
            |ad, x, y| {
                let xy = ad.mul(x, y);
                ad.exp(xy)
            },
            &[(0.5, 0.3), (1.0, 0.5)],
            1e-10,
        );
    }

    #[test]
    fn test_reverse_vs_forward_complex() {
        // f = ln(x² + y²)
        assert_reverse_matches_forward(
            |ad, x, y| {
                let x2 = ad.square(x);
                let y2 = ad.square(y);
                let sum = ad.add(x2, y2);
                ad.ln(sum)
            },
            &[(1.0, 1.0), (2.0, 3.0), (0.5, 1.5)],
            1e-10,
        );
    }

    // =========================================================================
    // Per-operation gradient tests (1D, reverse mode)
    // =========================================================================

    /// Helper for single-input per-operation gradient tests.
    fn assert_gradient_1d(
        build_fn: impl FnOnce(&mut AutoDiff, Var) -> Var,
        points: &[f64],
        expected_fn: impl Fn(f64) -> f64,
        epsilon: f64,
    ) {
        let mut ad = AutoDiff::new();
        let x = ad.var(points[0]).unwrap();
        let f = build_fn(&mut ad, x);

        let mut cg = ad.compile_primal(f, &[x]).unwrap();
        for &xv in points {
            cg.eval(&[xv]).unwrap();
            let grad = cg.gradient();
            assert_relative_eq!(grad[0], expected_fn(xv), epsilon = epsilon);
        }
    }

    #[test]
    fn test_reverse_gradient_neg() {
        assert_gradient_1d(|ad, x| ad.neg(x), &[1.0, -2.0, 0.0], |_| -1.0, 1e-12);
    }

    #[test]
    fn test_reverse_gradient_sin() {
        assert_gradient_1d(
            |ad, x| ad.sin(x),
            &[0.0, 0.5, 1.0, 2.0],
            |x| x.cos(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_cos() {
        assert_gradient_1d(
            |ad, x| ad.cos(x),
            &[0.0, 0.5, 1.0, 2.0],
            |x| -x.sin(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_tan() {
        assert_gradient_1d(
            |ad, x| ad.tan(x),
            &[0.0, 0.3, 0.7],
            |x| 1.0 / (x.cos() * x.cos()),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_exp() {
        assert_gradient_1d(
            |ad, x| ad.exp(x),
            &[0.0, 1.0, -1.0, 2.0],
            |x| x.exp(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_ln() {
        assert_gradient_1d(
            |ad, x| ad.ln(x),
            &[0.5, 1.0, 2.0, 10.0],
            |x| 1.0 / x,
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_sqrt() {
        assert_gradient_1d(
            |ad, x| ad.sqrt(x),
            &[1.0, 4.0, 9.0],
            |x| 0.5 / x.sqrt(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_sinh() {
        assert_gradient_1d(
            |ad, x| ad.sinh(x),
            &[0.0, 1.0, -1.0],
            |x| x.cosh(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_cosh() {
        assert_gradient_1d(
            |ad, x| ad.cosh(x),
            &[0.0, 1.0, -1.0],
            |x| x.sinh(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_tanh() {
        assert_gradient_1d(
            |ad, x| ad.tanh(x),
            &[0.0, 0.5, -0.5],
            |x| 1.0 - x.tanh().powi(2),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_asin() {
        assert_gradient_1d(
            |ad, x| ad.asin(x),
            &[0.0, 0.3, 0.5, -0.3],
            |x| 1.0 / (1.0 - x * x).sqrt(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_acos() {
        assert_gradient_1d(
            |ad, x| ad.acos(x),
            &[0.0, 0.3, 0.5, -0.3],
            |x| -1.0 / (1.0 - x * x).sqrt(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_atan() {
        assert_gradient_1d(
            |ad, x| ad.atan(x),
            &[0.0, 1.0, -1.0, 2.0],
            |x| 1.0 / (1.0 + x * x),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_asinh() {
        assert_gradient_1d(
            |ad, x| ad.asinh(x),
            &[0.0, 1.0, -1.0, 2.0],
            |x| 1.0 / (x * x + 1.0).sqrt(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_acosh() {
        assert_gradient_1d(
            |ad, x| ad.acosh(x),
            &[1.5, 2.0, 3.0],
            |x| 1.0 / (x * x - 1.0).sqrt(),
            1e-10,
        );
    }

    #[test]
    fn test_reverse_gradient_atanh() {
        assert_gradient_1d(
            |ad, x| ad.atanh(x),
            &[0.0, 0.3, -0.3, 0.5],
            |x| 1.0 / (1.0 - x * x),
            1e-10,
        );
    }

    // Binary ops as 2D gradient tests

    #[test]
    fn test_reverse_gradient_add_2d() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.add(x, y);

        let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
        let grad = cg.eval_gradient(&[2.0, 3.0]).unwrap();
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(grad[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_reverse_gradient_sub_2d() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.sub(x, y);

        let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
        let grad = cg.eval_gradient(&[2.0, 3.0]).unwrap();
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(grad[1], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_reverse_gradient_mul_2d() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.mul(x, y);

        let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
        let grad = cg.eval_gradient(&[2.0, 3.0]).unwrap();
        assert_relative_eq!(grad[0], 3.0, epsilon = 1e-12); // df/dx = y
        assert_relative_eq!(grad[1], 2.0, epsilon = 1e-12); // df/dy = x
    }

    #[test]
    fn test_reverse_gradient_div_2d() {
        let mut ad = AutoDiff::new();
        let x = ad.var(6.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.div(x, y);

        let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
        let grad = cg.eval_gradient(&[6.0, 3.0]).unwrap();
        assert_relative_eq!(grad[0], 1.0 / 3.0, epsilon = 1e-12);    // df/dx = 1/y
        assert_relative_eq!(grad[1], -6.0 / 9.0, epsilon = 1e-12);   // df/dy = -x/y²
    }

    #[test]
    fn test_reverse_gradient_pow_2d() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.pow(x, y);

        let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
        let grad = cg.eval_gradient(&[2.0, 3.0]).unwrap();
        // df/dx = y * x^(y-1) = 3 * 4 = 12
        assert_relative_eq!(grad[0], 12.0, epsilon = 1e-10);
        // df/dy = x^y * ln(x) = 8 * ln(2)
        assert_relative_eq!(grad[1], 8.0 * 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_compile_primal_matches_compile() {
        // Verify compile_primal gives same value as compile
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.mul(x, y);

        let mut cg_primal = ad.compile_primal(f, &[x, y]).unwrap();
        let mut cg_full = ad.compile(f, &[x, y], &[]).unwrap();

        cg_primal.eval(&[2.0, 3.0]).unwrap();
        cg_full.eval(&[2.0, 3.0]).unwrap();
        assert_eq!(cg_primal.value(), cg_full.value());
    }
}
