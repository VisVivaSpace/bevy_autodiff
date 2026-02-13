//! AutoDiff context - the main API for building computation graphs.

use bevy_ecs::world::World;
use bevy_entity_ptr::EntityHandle;

use crate::components::{
    BinaryInputs, BinaryOp, Dependencies, IsConstant, IsInput, UnaryInput, UnaryOp, Value,
    Variable,
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
}
