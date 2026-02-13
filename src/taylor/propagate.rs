//! Taylor coefficient propagation through the computation graph.

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

use crate::components::{
    BinaryInputs, Direction, IsConstant, IsInput, TaylorData, UnaryInput, Value,
};
use crate::context::{BinaryOpMarker, UnaryOpMarker};
use crate::graph::topological_order;
use crate::taylor::polynomial::{
    add_taylor, constant_taylor, div_taylor, identity_taylor, mul_taylor, neg_taylor, sub_taylor,
    TaylorCoeffs,
};
use crate::BinaryOp;
use crate::UnaryOp;

/// Propagates Taylor coefficients through the graph for a given direction.
///
/// This computes Taylor coefficients for all variables in the graph up to the
/// given order, starting from input variables (which have identity Taylor series
/// with direction components) and propagating through operations.
///
/// Returns the Taylor coefficients for the output variable.
pub fn propagate_taylor(
    world: &mut World,
    output: Entity,
    direction: &Direction,
    order: usize,
) -> Vec<f64> {
    // Get topological order
    let topo_order = topological_order(world, output);

    // Propagate coefficients
    for &entity in &topo_order {
        let coeffs = compute_taylor_coeffs(world, entity, direction, order);
        update_taylor_data(world, entity, direction, coeffs.to_vec());
    }

    // Return output coefficients
    get_coefficients(world, output, direction).unwrap_or_else(|| constant_taylor(0.0, order).to_vec())
}

/// Computes Taylor coefficients for a single entity given its inputs' coefficients.
pub fn compute_taylor_coeffs(
    world: &World,
    entity: Entity,
    direction: &Direction,
    order: usize,
) -> TaylorCoeffs {
    let entity_ref = world.entity(entity);

    // Input variables: identity Taylor series
    if entity_ref.contains::<IsInput>() {
        let value = entity_ref.get::<Value>().map(|v| v.get()).unwrap_or(0.0);
        // Get direction component for this input's index
        // For simplicity, we use the index based on input order
        // The direction component determines if this input contributes to the derivative
        let dir_component = get_input_direction_component(world, entity, direction);
        return identity_taylor(value, dir_component, order);
    }

    // Constants: constant Taylor series (all derivatives zero)
    if entity_ref.contains::<IsConstant>() {
        let value = entity_ref.get::<Value>().map(|v| v.get()).unwrap_or(0.0);
        return constant_taylor(value, order);
    }

    // Unary operation
    if let Some(op_marker) = entity_ref.get::<UnaryOpMarker>() {
        let input_handle = entity_ref.get::<UnaryInput>().expect("UnaryOp missing input");
        let input_coeffs = get_coefficients(world, input_handle.get().entity(), direction)
            .expect("Input coefficients not computed");

        return compute_unary_taylor(op_marker.0, &input_coeffs, order);
    }

    // Binary operation
    if let Some(op_marker) = entity_ref.get::<BinaryOpMarker>() {
        let inputs = entity_ref
            .get::<BinaryInputs>()
            .expect("BinaryOp missing inputs");
        let left_coeffs = get_coefficients(world, inputs.left.entity(), direction)
            .expect("Left input coefficients not computed");
        let right_coeffs = get_coefficients(world, inputs.right.entity(), direction)
            .expect("Right input coefficients not computed");

        return compute_binary_taylor(op_marker.0, &left_coeffs, &right_coeffs, order);
    }

    // Unknown entity type - treat as constant
    let value = entity_ref.get::<Value>().map(|v| v.get()).unwrap_or(0.0);
    constant_taylor(value, order)
}

/// Computes Taylor coefficients for a unary operation.
pub(crate) fn compute_unary_taylor(op: UnaryOp, input: &[f64], order: usize) -> TaylorCoeffs {
    use crate::taylor::rules::{
        acos_taylor, acosh_taylor, asin_taylor, asinh_taylor, atan_taylor, atanh_taylor,
        cos_taylor, cosh_taylor, exp_taylor, ln_taylor, sin_taylor, sinh_taylor, sqrt_taylor,
        tan_taylor, tanh_taylor,
    };

    match op {
        UnaryOp::Neg => neg_taylor(input, order),
        UnaryOp::Sin => sin_taylor(input, order).into(),
        UnaryOp::Cos => cos_taylor(input, order).into(),
        UnaryOp::Tan => tan_taylor(input, order),
        UnaryOp::Exp => exp_taylor(input, order).into(),
        UnaryOp::Ln => ln_taylor(input, order).expect("ln domain error"),
        UnaryOp::Sqrt => sqrt_taylor(input, order).expect("sqrt domain error"),
        UnaryOp::Sinh => sinh_taylor(input, order).into(),
        UnaryOp::Cosh => cosh_taylor(input, order).into(),
        UnaryOp::Tanh => tanh_taylor(input, order),
        UnaryOp::Asin => asin_taylor(input, order).expect("asin domain error"),
        UnaryOp::Acos => acos_taylor(input, order).expect("acos domain error"),
        UnaryOp::Atan => atan_taylor(input, order),
        UnaryOp::Asinh => asinh_taylor(input, order),
        UnaryOp::Acosh => acosh_taylor(input, order).expect("acosh domain error"),
        UnaryOp::Atanh => atanh_taylor(input, order).expect("atanh domain error"),
    }
}

/// Computes Taylor coefficients for a binary operation.
pub(crate) fn compute_binary_taylor(op: BinaryOp, left: &[f64], right: &[f64], order: usize) -> TaylorCoeffs {
    use crate::taylor::rules::pow_taylor;

    match op {
        BinaryOp::Add => add_taylor(left, right, order),
        BinaryOp::Sub => sub_taylor(left, right, order),
        BinaryOp::Mul => mul_taylor(left, right, order),
        BinaryOp::Div => div_taylor(left, right, order).expect("division by zero"),
        BinaryOp::Pow => pow_taylor(left, right, order).into(),
    }
}

/// Gets the direction component for an input variable.
///
/// This determines how much this input contributes to the directional derivative.
/// Returns direction[input_index] where input_index is the order in which
/// this input was created.
fn get_input_direction_component(world: &World, entity: Entity, direction: &Direction) -> f64 {
    // We need to find which input index this entity corresponds to
    // This is stored in the Dependencies component during var creation
    let entity_ref = world.entity(entity);
    if let Some(deps) = entity_ref.get::<crate::components::Dependencies>() {
        // Find the set bit position (input index)
        if deps.mask != 0 {
            let input_index = deps.mask.trailing_zeros() as usize;
            return direction.get(input_index) as f64;
        }
    }
    0.0
}

/// Gets the Taylor coefficients for an entity in a given direction.
fn get_coefficients(world: &World, entity: Entity, direction: &Direction) -> Option<Vec<f64>> {
    world
        .entity(entity)
        .get::<TaylorData>()
        .and_then(|data| data.get_directional(direction).cloned())
}

/// Updates the Taylor data for an entity with new directional coefficients.
fn update_taylor_data(world: &mut World, entity: Entity, direction: &Direction, coeffs: Vec<f64>) {
    if let Some(mut taylor_data) = world.entity_mut(entity).get_mut::<TaylorData>() {
        taylor_data.set_directional(direction.clone(), coeffs);
    } else {
        // Create TaylorData if it doesn't exist
        let mut data = TaylorData::new();
        data.set_directional(direction.clone(), coeffs);
        world.entity_mut(entity).insert(data);
    }
}

/// Extracts the k-th derivative from Taylor coefficients.
///
/// Since coefficients are stored normalized (divided by k!), we multiply by k!
/// to get the actual derivative value.
pub fn extract_derivative(coeffs: &[f64], k: usize) -> f64 {
    coeffs.get(k).copied().unwrap_or(0.0) * crate::util::factorial(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    #[test]
    fn test_propagate_input() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);

        let direction = Direction::basis(1, 0);
        let coeffs = propagate_taylor(ad.world_mut(), x.entity(), &direction, 2);

        // Input: f(t) = 5 + 1*t = [5, 1, 0]
        assert_eq!(coeffs, vec![5.0, 1.0, 0.0]);
    }

    #[test]
    fn test_propagate_constant() {
        let mut ad = AutoDiff::new();
        let c = ad.constant(3.0);

        let direction = Direction::basis(0, 0);
        let coeffs = propagate_taylor(ad.world_mut(), c.entity(), &direction, 2);

        // Constant: f(t) = 3 = [3, 0, 0]
        assert_eq!(coeffs, vec![3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_propagate_add() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let z = ad.add(x, y);

        // Direction: differentiate w.r.t. both x and y
        let direction = Direction::new(vec![1, 1]);
        let coeffs = propagate_taylor(ad.world_mut(), z.entity(), &direction, 2);

        // x(t) = 2 + t, y(t) = 3 + t
        // z(t) = x(t) + y(t) = 5 + 2t
        assert_eq!(coeffs, vec![5.0, 2.0, 0.0]);
    }

    #[test]
    fn test_propagate_sub() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let y = ad.var(3.0);
        let z = ad.sub(x, y);

        let direction = Direction::new(vec![1, 1]);
        let coeffs = propagate_taylor(ad.world_mut(), z.entity(), &direction, 2);

        // x(t) = 5 + t, y(t) = 3 + t
        // z(t) = x(t) - y(t) = 2 + 0*t
        assert_eq!(coeffs, vec![2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_propagate_mul() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let z = ad.mul(x, y);

        // Direction: only differentiate w.r.t. x
        let direction = Direction::new(vec![1, 0]);
        let coeffs = propagate_taylor(ad.world_mut(), z.entity(), &direction, 2);

        // x(t) = 2 + t, y(t) = 3 + 0*t = 3
        // z(t) = (2 + t) * 3 = 6 + 3t
        assert_eq!(coeffs, vec![6.0, 3.0, 0.0]);
    }

    #[test]
    fn test_propagate_div() {
        let mut ad = AutoDiff::new();
        let x = ad.var(6.0);
        let y = ad.var(2.0);
        let z = ad.div(x, y);

        // Direction: only differentiate w.r.t. x
        let direction = Direction::new(vec![1, 0]);
        let coeffs = propagate_taylor(ad.world_mut(), z.entity(), &direction, 2);

        // x(t) = 6 + t, y(t) = 2
        // z(t) = (6 + t) / 2 = 3 + 0.5*t
        assert_relative_eq!(coeffs[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_propagate_neg() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let y = ad.neg(x);

        let direction = Direction::basis(1, 0);
        let coeffs = propagate_taylor(ad.world_mut(), y.entity(), &direction, 2);

        // x(t) = 5 + t
        // -x(t) = -5 - t
        assert_eq!(coeffs, vec![-5.0, -1.0, 0.0]);
    }

    #[test]
    fn test_propagate_x_squared() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.square(x);

        let direction = Direction::basis(1, 0);
        let coeffs = propagate_taylor(ad.world_mut(), y.entity(), &direction, 3);

        // x(t) = 2 + t
        // x²(t) = (2 + t)² = 4 + 4t + t²
        assert_eq!(coeffs[0], 4.0); // f(2) = 4
        assert_eq!(coeffs[1], 4.0); // f'(2)/1! = 4
        assert_eq!(coeffs[2], 1.0); // f''(2)/2! = 1

        // Extract derivatives
        assert_eq!(extract_derivative(&coeffs, 0), 4.0); // value
        assert_eq!(extract_derivative(&coeffs, 1), 4.0); // first derivative
        assert_eq!(extract_derivative(&coeffs, 2), 2.0); // second derivative: 2
    }

    #[test]
    fn test_propagate_x_cubed() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let x2 = ad.square(x);
        let x3 = ad.mul(x2, x);

        let direction = Direction::basis(1, 0);
        let coeffs = propagate_taylor(ad.world_mut(), x3.entity(), &direction, 4);

        // x(t) = 2 + t
        // x³(t) = (2 + t)³ = 8 + 12t + 6t² + t³
        // These are normalized coefficients: coeff[k] = f^(k)(2) / k!
        assert_eq!(coeffs[0], 8.0); // f(2) = 8
        assert_eq!(coeffs[1], 12.0); // f'(2)/1! = 12 (f' = 3x² = 12 at x=2)
        assert_eq!(coeffs[2], 6.0); // f''(2)/2! = 12/2 = 6 (f'' = 6x = 12 at x=2)
        assert_relative_eq!(coeffs[3], 1.0, epsilon = 1e-10); // f'''(2)/3! = 6/6 = 1

        // Extract third derivative: d³/dx³(x³) = 6
        assert_relative_eq!(extract_derivative(&coeffs, 3), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_propagate_product_rule() {
        // f(x, y) = x * y
        // ∂f/∂x = y, ∂f/∂y = x
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let y = ad.var(5.0);
        let f = ad.mul(x, y);

        // Differentiate w.r.t. x only
        let dir_x = Direction::new(vec![1, 0]);
        let coeffs_x = propagate_taylor(ad.world_mut(), f.entity(), &dir_x, 1);
        assert_eq!(extract_derivative(&coeffs_x, 1), 5.0); // ∂f/∂x = y = 5

        // Differentiate w.r.t. y only
        let dir_y = Direction::new(vec![0, 1]);
        let coeffs_y = propagate_taylor(ad.world_mut(), f.entity(), &dir_y, 1);
        assert_eq!(extract_derivative(&coeffs_y, 1), 3.0); // ∂f/∂y = x = 3
    }

    #[test]
    fn test_propagate_chain_rule() {
        // f(x) = (x + 1)²
        // f'(x) = 2(x + 1)
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let one = ad.constant(1.0);
        let x_plus_1 = ad.add(x, one);
        let f = ad.square(x_plus_1);

        let direction = Direction::basis(1, 0);
        let coeffs = propagate_taylor(ad.world_mut(), f.entity(), &direction, 2);

        // At x = 2: f(2) = 9, f'(2) = 6, f''(2) = 2
        assert_eq!(coeffs[0], 9.0);
        assert_eq!(extract_derivative(&coeffs, 1), 6.0);
        assert_eq!(extract_derivative(&coeffs, 2), 2.0);
    }

    #[test]
    fn test_propagate_quotient() {
        // f(x) = 1/x
        // f'(x) = -1/x²
        // f''(x) = 2/x³
        let mut ad = AutoDiff::new();
        let one = ad.constant(1.0);
        let x = ad.var(2.0);
        let f = ad.div(one, x);

        let direction = Direction::basis(1, 0);
        let coeffs = propagate_taylor(ad.world_mut(), f.entity(), &direction, 3);

        // At x = 2: f(2) = 0.5, f'(2) = -0.25, f''(2) = 0.25
        assert_relative_eq!(coeffs[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(extract_derivative(&coeffs, 1), -0.25, epsilon = 1e-10);
        assert_relative_eq!(extract_derivative(&coeffs, 2), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_derivative() {
        let coeffs = vec![1.0, 2.0, 3.0, 4.0];

        assert_eq!(extract_derivative(&coeffs, 0), 1.0); // 1.0 * 0! = 1.0
        assert_eq!(extract_derivative(&coeffs, 1), 2.0); // 2.0 * 1! = 2.0
        assert_eq!(extract_derivative(&coeffs, 2), 6.0); // 3.0 * 2! = 6.0
        assert_eq!(extract_derivative(&coeffs, 3), 24.0); // 4.0 * 3! = 24.0
    }
}
