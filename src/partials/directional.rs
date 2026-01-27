//! Directional derivative extraction.

use crate::components::Direction;
use crate::taylor::propagate::{extract_derivative, propagate_taylor};
use crate::var::Var;
use bevy_ecs::world::World;

/// Extracts the k-th directional derivative from Taylor coefficients.
///
/// Given direction d, computes D_d^k f where:
/// D_d f = d₁·∂f/∂x₁ + d₂·∂f/∂x₂ + ... + dₙ·∂f/∂xₙ
///
/// # Arguments
/// - `world`: The ECS world
/// - `output`: The variable to differentiate
/// - `direction`: The direction vector
/// - `order`: The order of the directional derivative
///
/// # Returns
/// D_d^order f at the current input values
pub fn extract_directional_derivative(
    world: &mut World,
    output: Var,
    direction: &Direction,
    order: usize,
) -> f64 {
    let coeffs = propagate_taylor(world, output.entity(), direction, order);
    extract_derivative(&coeffs, order)
}

/// Computes a pure partial derivative: ∂^n f / ∂xᵢ^n
///
/// This is simply the n-th directional derivative in direction eᵢ.
///
/// # Arguments
/// - `world`: The ECS world
/// - `output`: The variable to differentiate
/// - `var_index`: Which input variable (0-indexed)
/// - `order`: The derivative order
/// - `num_inputs`: Total number of input variables
///
/// # Returns
/// ∂^n f / ∂xᵢ^n at the current input values
pub fn get_pure_partial(
    world: &mut World,
    output: Var,
    var_index: usize,
    order: usize,
    num_inputs: usize,
) -> f64 {
    let direction = Direction::basis(num_inputs, var_index);
    extract_directional_derivative(world, output, &direction, order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    #[test]
    fn test_extract_directional_x_squared() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.square(x); // y = x²

        // Direction (1) = d/dx
        let direction = Direction::basis(1, 0);

        // First directional derivative = dy/dx = 2x = 4
        let d1 = extract_directional_derivative(ad.world_mut(), y, &direction, 1);
        assert_eq!(d1, 4.0);

        // Second directional derivative = d²y/dx² = 2
        let d2 = extract_directional_derivative(ad.world_mut(), y, &direction, 2);
        assert_eq!(d2, 2.0);
    }

    #[test]
    fn test_get_pure_partial() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let y = ad.square(x); // y = x²

        // ∂y/∂x = 2x = 6
        let partial1 = get_pure_partial(ad.world_mut(), y, 0, 1, 1);
        assert_eq!(partial1, 6.0);

        // ∂²y/∂x² = 2
        let partial2 = get_pure_partial(ad.world_mut(), y, 0, 2, 1);
        assert_eq!(partial2, 2.0);
    }

    #[test]
    fn test_directional_derivative_multivariate() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);

        // f = x*y
        let f = ad.mul(x, y);

        // Directional derivative in direction (1, 1):
        // D_{(1,1)} f = ∂f/∂x + ∂f/∂y = y + x = 2 + 1 = 3
        let direction = Direction::new(vec![1, 1]);
        let d1 = extract_directional_derivative(ad.world_mut(), f, &direction, 1);
        assert_eq!(d1, 3.0);
    }

    #[test]
    fn test_pure_partials_multivariate() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // f = x² + y²
        let x2 = ad.square(x);
        let y2 = ad.square(y);
        let f = ad.add(x2, y2);

        // ∂f/∂x = 2x = 4
        assert_eq!(get_pure_partial(ad.world_mut(), f, 0, 1, 2), 4.0);

        // ∂f/∂y = 2y = 6
        assert_eq!(get_pure_partial(ad.world_mut(), f, 1, 1, 2), 6.0);

        // ∂²f/∂x² = 2
        assert_eq!(get_pure_partial(ad.world_mut(), f, 0, 2, 2), 2.0);

        // ∂²f/∂y² = 2
        assert_eq!(get_pure_partial(ad.world_mut(), f, 1, 2, 2), 2.0);
    }

    #[test]
    fn test_exp_derivatives() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let f = ad.exp(x); // e^x at x=0

        // All derivatives of e^x at x=0 equal 1
        for order in 0..=5 {
            let partial = get_pure_partial(ad.world_mut(), f, 0, order, 1);
            assert_relative_eq!(partial, 1.0, epsilon = 1e-10);
        }
    }
}
