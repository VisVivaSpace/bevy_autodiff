//! Operation components defining how variables are computed.

use bevy_ecs::component::Component;
use bevy_entity_ptr::EntityHandle;

/// Unary operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryOp {
    /// Negation: -x
    Neg,
    /// Sine: sin(x)
    Sin,
    /// Cosine: cos(x)
    Cos,
    /// Tangent: tan(x)
    Tan,
    /// Exponential: e^x
    Exp,
    /// Natural logarithm: ln(x)
    Ln,
    /// Square root: √x
    Sqrt,
    /// Hyperbolic sine: sinh(x)
    Sinh,
    /// Hyperbolic cosine: cosh(x)
    Cosh,
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// Inverse sine: asin(x)
    Asin,
    /// Inverse cosine: acos(x)
    Acos,
    /// Inverse tangent: atan(x)
    Atan,
    /// Inverse hyperbolic sine: asinh(x)
    Asinh,
    /// Inverse hyperbolic cosine: acosh(x)
    Acosh,
    /// Inverse hyperbolic tangent: atanh(x)
    Atanh,
}

impl UnaryOp {
    /// Returns the name of the operation for debugging.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Exp => "exp",
            Self::Ln => "ln",
            Self::Sqrt => "sqrt",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
        }
    }
}

/// Binary operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryOp {
    /// Addition: x + y
    Add,
    /// Subtraction: x - y
    Sub,
    /// Multiplication: x * y
    Mul,
    /// Division: x / y
    Div,
    /// Power: x^y
    Pow,
    /// Power with logarithmic differentiation: x^y
    ///
    /// Primal evaluation is identical to [`Pow`](BinaryOp::Pow), but symbolic
    /// differentiation uses the logarithmic form `d(a^b)/da = a^b · b · (da/a)`
    /// instead of the standard power rule `b · a^(b-1) · da`. This avoids
    /// catastrophic cancellation in f32 for second-order and higher derivatives.
    ///
    /// **Requirement:** base must be positive (`a > 0`). Produces NaN otherwise.
    PowLog,
    /// Division with logarithmic differentiation: x / y
    ///
    /// Primal evaluation is identical to [`Div`](BinaryOp::Div), but symbolic
    /// differentiation uses the logarithmic form `d(a/b) = (a/b) · (da/a - db/b)`
    /// instead of the quotient rule `(da·b - a·db) / b²`. This avoids
    /// catastrophic cancellation in f32 for second-order and higher derivatives.
    ///
    /// **Requirement:** both operands must be nonzero. Produces NaN otherwise.
    DivLog,
}

impl BinaryOp {
    /// Returns the name of the operation for debugging.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Pow => "pow",
            Self::PowLog => "pow_log",
            Self::DivLog => "div_log",
        }
    }

    /// Returns the mathematical symbol for the operation.
    pub const fn symbol(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div | Self::DivLog => "/",
            Self::Pow | Self::PowLog => "^",
        }
    }
}

/// Component marker for unary operations (stores the operation type).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnaryOpMarker(pub(crate) UnaryOp);

impl UnaryOpMarker {
    /// Returns the operation type.
    #[inline]
    pub fn op(&self) -> UnaryOp {
        self.0
    }
}

/// Component marker for binary operations (stores the operation type).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryOpMarker(pub(crate) BinaryOp);

impl BinaryOpMarker {
    /// Returns the operation type.
    #[inline]
    pub fn op(&self) -> BinaryOp {
        self.0
    }
}

/// Component storing the input entity for a unary operation.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnaryInput(pub EntityHandle);

impl UnaryInput {
    /// Creates a new UnaryInput component.
    #[inline]
    pub const fn new(input: EntityHandle) -> Self {
        Self(input)
    }

    /// Returns the input entity handle.
    #[inline]
    pub const fn get(&self) -> EntityHandle {
        self.0
    }
}

/// Component storing the input entities for a binary operation.
/// The order matters for non-commutative operations (Sub, Div, Pow, DivLog, PowLog).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryInputs {
    /// Left operand (first argument)
    pub left: EntityHandle,
    /// Right operand (second argument)
    pub right: EntityHandle,
}

impl BinaryInputs {
    /// Creates a new BinaryInputs component.
    #[inline]
    pub const fn new(left: EntityHandle, right: EntityHandle) -> Self {
        Self { left, right }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;

    #[test]
    fn test_unary_op_names() {
        assert_eq!(UnaryOp::Neg.name(), "neg");
        assert_eq!(UnaryOp::Sin.name(), "sin");
        assert_eq!(UnaryOp::Cos.name(), "cos");
        assert_eq!(UnaryOp::Tan.name(), "tan");
        assert_eq!(UnaryOp::Exp.name(), "exp");
        assert_eq!(UnaryOp::Ln.name(), "ln");
        assert_eq!(UnaryOp::Sqrt.name(), "sqrt");
        assert_eq!(UnaryOp::Sinh.name(), "sinh");
        assert_eq!(UnaryOp::Cosh.name(), "cosh");
        assert_eq!(UnaryOp::Tanh.name(), "tanh");
        assert_eq!(UnaryOp::Asin.name(), "asin");
        assert_eq!(UnaryOp::Acos.name(), "acos");
        assert_eq!(UnaryOp::Atan.name(), "atan");
        assert_eq!(UnaryOp::Asinh.name(), "asinh");
        assert_eq!(UnaryOp::Acosh.name(), "acosh");
        assert_eq!(UnaryOp::Atanh.name(), "atanh");
    }

    #[test]
    fn test_binary_op_names_and_symbols() {
        assert_eq!(BinaryOp::Add.name(), "add");
        assert_eq!(BinaryOp::Add.symbol(), "+");
        assert_eq!(BinaryOp::Sub.name(), "sub");
        assert_eq!(BinaryOp::Sub.symbol(), "-");
        assert_eq!(BinaryOp::Mul.name(), "mul");
        assert_eq!(BinaryOp::Mul.symbol(), "*");
        assert_eq!(BinaryOp::Div.name(), "div");
        assert_eq!(BinaryOp::Div.symbol(), "/");
        assert_eq!(BinaryOp::Pow.name(), "pow");
        assert_eq!(BinaryOp::Pow.symbol(), "^");
    }

    #[test]
    fn test_unary_input_component() {
        let mut world = World::new();
        let input = world.spawn_empty().id();
        let input_handle = EntityHandle::new(input);

        let entity = world.spawn(UnaryInput::new(input_handle)).id();
        let unary_input = world.entity(entity).get::<UnaryInput>().unwrap();
        assert_eq!(unary_input.get(), input_handle);
    }

    #[test]
    fn test_binary_inputs_component() {
        let mut world = World::new();
        let left = world.spawn_empty().id();
        let right = world.spawn_empty().id();
        let left_handle = EntityHandle::new(left);
        let right_handle = EntityHandle::new(right);

        let entity = world
            .spawn(BinaryInputs::new(left_handle, right_handle))
            .id();
        let binary_inputs = world.entity(entity).get::<BinaryInputs>().unwrap();
        assert_eq!(binary_inputs.left, left_handle);
        assert_eq!(binary_inputs.right, right_handle);
    }

    #[test]
    fn test_operation_markers() {
        use super::{BinaryOpMarker, UnaryOpMarker};
        let mut world = World::new();

        let unary = world.spawn(UnaryOpMarker(UnaryOp::Sin)).id();
        let binary = world.spawn(BinaryOpMarker(BinaryOp::Mul)).id();

        assert_eq!(
            world.entity(unary).get::<UnaryOpMarker>().unwrap().0,
            UnaryOp::Sin
        );
        assert_eq!(
            world.entity(binary).get::<BinaryOpMarker>().unwrap().0,
            BinaryOp::Mul
        );
    }
}
