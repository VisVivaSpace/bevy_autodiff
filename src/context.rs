//! AutoDiff context - the main API for building computation graphs.

use std::collections::HashMap;

use bevy_ecs::world::World;
use bevy_entity_ptr::EntityHandle;

use crate::components::{
    BinaryInputs, BinaryOp, Dependencies, IsConstant, IsInput, UnaryInput, UnaryOp, Value,
    Variable,
};
use crate::graph::topology::topological_order;
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
    /// Input variables in creation order.
    inputs: Vec<Var>,
}

impl AutoDiff {
    /// Creates a new, empty autodiff context.
    #[inline]
    pub fn new() -> Self {
        Self {
            world: World::new(),
            input_count: 0,
            inputs: Vec::new(),
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
            ))
            .id();

        let v = Var::new(entity);
        self.inputs.push(v);
        v
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

    /// Sets multiple input values at once.
    ///
    /// # Panics
    /// Panics if any variable is not an input variable.
    pub fn set_inputs(&mut self, inputs: &[(Var, f64)]) {
        for &(var, value) in inputs {
            self.set_input(var, value);
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
    /// let x = ad.var(2.0);
    /// let y = ad.var(3.0);
    /// let f = ad.mul(x, y); // f = x * y
    ///
    /// // First derivative: df/dx = y
    /// let dfdx = ad.differentiate(f, x);
    /// assert_eq!(ad.eval(dfdx), 3.0);
    ///
    /// // Mixed partial: d²f/dxdy = 1
    /// let d2fdxdy = ad.differentiate(dfdx, y);
    /// assert_eq!(ad.eval(d2fdxdy), 1.0);
    /// ```
    pub fn differentiate(&mut self, output: Var, wrt: Var) -> Var {
        let order = topological_order(&self.world, output.entity());
        let mut derivs: HashMap<bevy_ecs::entity::Entity, Var> = HashMap::new();

        let zero = self.constant(0.0);
        let one = self.constant(1.0);

        for &entity in &order {
            // Extract all info from the entity before any mutations
            let is_wrt = entity == wrt.entity();
            let is_input = self.world.entity(entity).contains::<IsInput>();
            let is_const = self.world.entity(entity).contains::<IsConstant>();
            let unary_op = self
                .world
                .entity(entity)
                .get::<UnaryOpMarker>()
                .map(|m| m.0);
            let binary_op = self
                .world
                .entity(entity)
                .get::<BinaryOpMarker>()
                .map(|m| m.0);
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
                let a_entity = unary_input_entity.unwrap();
                let a = Var::new(a_entity);
                let da = derivs[&a_entity];

                let dz = self.differentiate_unary(op, z, a, da, one);
                derivs.insert(entity, dz);
            } else if let Some(op) = binary_op {
                let (a_entity, b_entity) = binary_input_entities.unwrap();
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

        derivs[&output.entity()]
    }

    /// Apply chain rule for a unary operation: z = op(a), given da = d(a)/d(wrt).
    fn differentiate_unary(
        &mut self,
        op: UnaryOp,
        z: Var,
        a: Var,
        da: Var,
        one: Var,
    ) -> Var {
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
                    let b_val = self.eval(b);
                    let b_minus_1 = self.constant(b_val - 1.0);
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
        }
    }

    // =========================================================================
    // Smart Helpers (constant folding during differentiation)
    // =========================================================================

    /// Check if a Var is a constant with a specific value.
    fn is_const_value(&self, v: Var, val: f64) -> bool {
        self.is_constant(v) && self.eval(v) == val
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
            return self.constant(self.eval(a) + self.eval(b));
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
            return self.constant(self.eval(a) - self.eval(b));
        }
        self.sub(a, b)
    }

    /// Multiply with constant folding: return 0 if either is 0, identity if 1.
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
            return self.constant(self.eval(a) * self.eval(b));
        }
        self.mul(a, b)
    }

    /// Negate with constant folding: skip if 0, fold if constant.
    fn smart_neg(&mut self, a: Var) -> Var {
        if self.is_const_value(a, 0.0) {
            return a;
        }
        if self.is_constant(a) {
            return self.constant(-self.eval(a));
        }
        self.neg(a)
    }

    /// Divide with constant folding: return 0 if numerator is 0, identity if denom is 1.
    fn smart_div(&mut self, a: Var, b: Var) -> Var {
        if self.is_const_value(a, 0.0) {
            return self.constant(0.0);
        }
        if self.is_const_value(b, 1.0) {
            return a;
        }
        if self.is_constant(a) && self.is_constant(b) {
            return self.constant(self.eval(a) / self.eval(b));
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
    fn test_inputs_tracking() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);
        assert_eq!(ad.inputs(), &[x, y]);
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

    // =========================================================================
    // Differentiation tests
    // =========================================================================

    #[test]
    fn test_diff_identity() {
        // d/dx(x) = 1
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let dxdx = ad.differentiate(x, x);
        assert_eq!(ad.eval(dxdx), 1.0);
    }

    #[test]
    fn test_diff_constant() {
        // d/dx(c) = 0
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let c = ad.constant(5.0);
        let dc = ad.differentiate(c, x);
        assert_eq!(ad.eval(dc), 0.0);
    }

    #[test]
    fn test_diff_other_input() {
        // d/dx(y) = 0
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let y = ad.var(5.0);
        let dy = ad.differentiate(y, x);
        assert_eq!(ad.eval(dy), 0.0);
    }

    #[test]
    fn test_diff_neg() {
        // d/dx(-x) = -1
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let f = ad.neg(x);
        let df = ad.differentiate(f, x);
        assert_eq!(ad.eval(df), -1.0);
    }

    #[test]
    fn test_diff_add() {
        // d/dx(x + c) = 1
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let c = ad.constant(5.0);
        let f = ad.add(x, c);
        let df = ad.differentiate(f, x);
        assert_eq!(ad.eval(df), 1.0);
    }

    #[test]
    fn test_diff_sub() {
        // d/dx(x - c) = 1
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let c = ad.constant(5.0);
        let f = ad.sub(x, c);
        let df = ad.differentiate(f, x);
        assert_eq!(ad.eval(df), 1.0);
    }

    #[test]
    fn test_diff_mul_by_constant() {
        // d/dx(c * x) = c
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let c = ad.constant(5.0);
        let f = ad.mul(c, x);
        let df = ad.differentiate(f, x);
        assert_eq!(ad.eval(df), 5.0);
    }

    #[test]
    fn test_diff_mul_two_vars() {
        // d/dx(x * y) = y
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y);
        let df = ad.differentiate(f, x);
        assert_eq!(ad.eval(df), 3.0);
    }

    #[test]
    fn test_diff_div_by_constant() {
        // d/dx(x / c) = 1/c
        let mut ad = AutoDiff::new();
        let x = ad.var(6.0);
        let c = ad.constant(3.0);
        let f = ad.div(x, c);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_square() {
        // d/dx(x²) = 2x at x=3 → 6
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let f = ad.square(x);
        let df = ad.differentiate(f, x);
        assert_eq!(ad.eval(df), 6.0);
    }

    #[test]
    fn test_diff_pow_const_exp() {
        // d/dx(x^3) = 3x² at x=2 → 12
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let f = ad.powi(x, 3);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_sin() {
        // d/dx(sin(x)) = cos(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.sin(x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 0.5_f64.cos(), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_cos() {
        // d/dx(cos(x)) = -sin(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.cos(x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), -(0.5_f64.sin()), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_tan() {
        // d/dx(tan(x)) = 1/cos²(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.tan(x);
        let df = ad.differentiate(f, x);
        let expected = 1.0 / (0.5_f64.cos().powi(2));
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_exp() {
        // d/dx(exp(x)) = exp(x) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.exp(x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 1.0_f64.exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_ln() {
        // d/dx(ln(x)) = 1/x at x=2
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let f = ad.ln(x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_sqrt() {
        // d/dx(sqrt(x)) = 1/(2*sqrt(x)) at x=4 → 0.25
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);
        let f = ad.sqrt(x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_sinh() {
        // d/dx(sinh(x)) = cosh(x) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.sinh(x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 1.0_f64.cosh(), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_cosh() {
        // d/dx(cosh(x)) = sinh(x) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.cosh(x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 1.0_f64.sinh(), epsilon = 1e-10);
    }

    #[test]
    fn test_diff_tanh() {
        // d/dx(tanh(x)) = 1 - tanh²(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.tanh(x);
        let df = ad.differentiate(f, x);
        let expected = 1.0 - 0.5_f64.tanh().powi(2);
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_asin() {
        // d/dx(asin(x)) = 1/sqrt(1-x²) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.asin(x);
        let df = ad.differentiate(f, x);
        let expected = 1.0 / (1.0 - 0.25_f64).sqrt();
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_acos() {
        // d/dx(acos(x)) = -1/sqrt(1-x²) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.acos(x);
        let df = ad.differentiate(f, x);
        let expected = -1.0 / (1.0 - 0.25_f64).sqrt();
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_atan() {
        // d/dx(atan(x)) = 1/(1+x²) at x=1 → 0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.atan(x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_asinh() {
        // d/dx(asinh(x)) = 1/sqrt(x²+1) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.asinh(x);
        let df = ad.differentiate(f, x);
        let expected = 1.0 / (2.0_f64).sqrt();
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_acosh() {
        // d/dx(acosh(x)) = 1/sqrt(x²-1) at x=2
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let f = ad.acosh(x);
        let df = ad.differentiate(f, x);
        let expected = 1.0 / (3.0_f64).sqrt();
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_atanh() {
        // d/dx(atanh(x)) = 1/(1-x²) at x=0.5 → 1/0.75
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.atanh(x);
        let df = ad.differentiate(f, x);
        let expected = 1.0 / (1.0 - 0.25);
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_chain_rule() {
        // d/dx(sin(x²)) = 2x * cos(x²) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let x2 = ad.square(x);
        let f = ad.sin(x2);
        let df = ad.differentiate(f, x);
        let expected = 2.0 * 1.0_f64.cos(); // 2*1 * cos(1)
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_product_rule() {
        // d/dx(x * sin(x)) = sin(x) + x * cos(x) at x=1
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let sin_x = ad.sin(x);
        let f = ad.mul(x, sin_x);
        let df = ad.differentiate(f, x);
        let expected = 1.0_f64.sin() + 1.0 * 1.0_f64.cos();
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_quotient_rule() {
        // d/dx(sin(x)/x) = (cos(x)*x - sin(x)) / x² at x=2
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let sin_x = ad.sin(x);
        let f = ad.div(sin_x, x);
        let df = ad.differentiate(f, x);
        let expected = (2.0_f64.cos() * 2.0 - 2.0_f64.sin()) / 4.0;
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_second_derivative() {
        // d²/dx²(x³) = 6x at x=2 → 12
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let f = ad.powi(x, 3);
        let df = ad.differentiate(f, x);
        let d2f = ad.differentiate(df, x);
        assert_relative_eq!(ad.eval(d2f), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_third_derivative() {
        // d³/dx³(x⁴) = 24x at x=1 → 24
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.powi(x, 4);
        let df = ad.differentiate(f, x);
        let d2f = ad.differentiate(df, x);
        let d3f = ad.differentiate(d2f, x);
        assert_relative_eq!(ad.eval(d3f), 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_mixed_partial() {
        // d²/dxdy(x * y) = 1
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y);
        let dfdx = ad.differentiate(f, x);
        let d2fdxdy = ad.differentiate(dfdx, y);
        assert_relative_eq!(ad.eval(d2fdxdy), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_mixed_partial_symmetry() {
        // d²f/dxdy = d²f/dydx for f = x² * y + y³
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);

        let x2 = ad.square(x);
        let x2y = ad.mul(x2, y);
        let y2 = ad.square(y);
        let y3 = ad.mul(y, y2);
        let f = ad.add(x2y, y3);

        let dfdx = ad.differentiate(f, x);
        let d2fdxdy = ad.differentiate(dfdx, y);

        let dfdy = ad.differentiate(f, y);
        let d2fdydx = ad.differentiate(dfdy, x);

        assert_relative_eq!(ad.eval(d2fdxdy), ad.eval(d2fdydx), epsilon = 1e-10);
        // d²f/dxdy = 2x at (1,2) → 2
        assert_relative_eq!(ad.eval(d2fdxdy), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_exp_chain() {
        // d/dx(exp(sin(x))) = exp(sin(x)) * cos(x) at x=0.5
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let sin_x = ad.sin(x);
        let f = ad.exp(sin_x);
        let df = ad.differentiate(f, x);
        let expected = (0.5_f64.sin()).exp() * 0.5_f64.cos();
        assert_relative_eq!(ad.eval(df), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_polynomial() {
        // f = x³ + 2x² + 3x + 4
        // f'= 3x² + 4x + 3
        // At x=2: f' = 12 + 8 + 3 = 23
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
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

        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 23.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_rosenbrock() {
        // f = (1-x)² + 100(y-x²)²
        // df/dx = -2(1-x) + 100 * 2(y-x²) * (-2x) = 2(x-1) - 400x(y-x²)
        // At (1,1): df/dx = 0
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(1.0);
        let one = ad.constant(1.0);
        let hundred = ad.constant(100.0);

        let one_minus_x = ad.sub(one, x);
        let term1 = ad.square(one_minus_x);
        let x_sq = ad.square(x);
        let y_minus_x_sq = ad.sub(y, x_sq);
        let term2_inner = ad.square(y_minus_x_sq);
        let term2 = ad.mul(hundred, term2_inner);
        let f = ad.add(term1, term2);

        let dfdx = ad.differentiate(f, x);
        let dfdy = ad.differentiate(f, y);

        assert_relative_eq!(ad.eval(dfdx), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(dfdy), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_rosenbrock_away_from_min() {
        // f = (1-x)² + 100(y-x²)²
        // df/dx = 2(x-1) - 400x(y-x²)
        // df/dy = 200(y-x²)
        // At (0,0): df/dx = -2, df/dy = 0
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let y = ad.var(0.0);
        let one = ad.constant(1.0);
        let hundred = ad.constant(100.0);

        let one_minus_x = ad.sub(one, x);
        let term1 = ad.square(one_minus_x);
        let x_sq = ad.square(x);
        let y_minus_x_sq = ad.sub(y, x_sq);
        let term2_inner = ad.square(y_minus_x_sq);
        let term2 = ad.mul(hundred, term2_inner);
        let f = ad.add(term1, term2);

        let dfdx = ad.differentiate(f, x);
        let dfdy = ad.differentiate(f, y);

        assert_relative_eq!(ad.eval(dfdx), -2.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(dfdy), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diff_constant_folding() {
        // Verify that differentiating a constant wrt x produces a
        // constant entity, not just a value of 0
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let c = ad.constant(5.0);
        let dc = ad.differentiate(c, x);
        assert!(ad.is_constant(dc));
        assert_eq!(ad.eval(dc), 0.0);
    }
}
