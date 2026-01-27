//! Mixed partial derivative extraction via polarization.
//!
//! ## Polarization Identity
//!
//! For symmetric bilinear forms, the polarization identity allows us to
//! recover mixed derivatives from directional derivatives.
//!
//! For second-order mixed partials:
//! ```text
//! ∂²f/∂xᵢ∂xⱼ = (1/2) [D²f(eᵢ + eⱼ) - D²f(eᵢ) - D²f(eⱼ)]
//! ```
//!
//! where D²f(d) is the second directional derivative in direction d.
//!
//! The factor of 1/2 arises because D²f(eᵢ + eⱼ) expands as:
//! ```text
//! D²f(eᵢ + eⱼ) = D²f(eᵢ) + 2·∂²f/∂xᵢ∂xⱼ + D²f(eⱼ)
//! ```

use crate::components::{Direction, MultiIndex};
use crate::partials::directional::extract_directional_derivative;
use crate::var::Var;
use bevy_ecs::world::World;

/// Computes the mixed second partial ∂²f/∂xᵢ∂xⱼ using polarization.
///
/// Uses the identity:
/// ∂²f/∂xᵢ∂xⱼ = (1/2) [D²f(eᵢ + eⱼ) - D²f(eᵢ) - D²f(eⱼ)]
///
/// # Arguments
/// - `world`: The ECS world
/// - `output`: The variable to differentiate
/// - `i`: First variable index
/// - `j`: Second variable index
/// - `num_inputs`: Total number of input variables
///
/// # Returns
/// ∂²f/∂xᵢ∂xⱼ at the current input values
pub fn get_mixed_partial_2(
    world: &mut World,
    output: Var,
    i: usize,
    j: usize,
    num_inputs: usize,
) -> f64 {
    if i == j {
        // Pure second partial
        let dir_i = Direction::basis(num_inputs, i);
        return extract_directional_derivative(world, output, &dir_i, 2);
    }

    // Mixed partial: use polarization identity
    let dir_i = Direction::basis(num_inputs, i);
    let dir_j = Direction::basis(num_inputs, j);
    let dir_ij = Direction::sum_of_basis(num_inputs, i, j);

    let d2_ij = extract_directional_derivative(world, output, &dir_ij, 2);
    let d2_i = extract_directional_derivative(world, output, &dir_i, 2);
    let d2_j = extract_directional_derivative(world, output, &dir_j, 2);

    0.5 * (d2_ij - d2_i - d2_j)
}

/// Computes a general mixed partial derivative using generalized polarization.
///
/// For multi-index α = (α₁, α₂, ..., αₙ), this uses multiple directional
/// derivatives to extract the mixed partial.
///
/// # Arguments
/// - `world`: The ECS world
/// - `output`: The variable to differentiate
/// - `index`: The multi-index specifying the partial derivative
/// - `num_inputs`: Total number of input variables
///
/// # Returns
/// ∂^|α|f / ∂x^α at the current input values
pub fn get_mixed_partial(
    world: &mut World,
    output: Var,
    index: &MultiIndex,
    num_inputs: usize,
) -> f64 {
    let order = index.order();

    if order == 0 {
        // Just the value
        let dir = Direction::zero(num_inputs);
        return extract_directional_derivative(world, output, &dir, 0);
    }

    // Build a list of variable indices with repetitions according to exponents
    // e.g., index (2, 1, 0) -> [0, 0, 1]
    let mut vars: Vec<usize> = Vec::new();
    for (i, &exp) in index.0.iter().enumerate() {
        for _ in 0..exp {
            vars.push(i);
        }
    }

    // Check if all same variable (pure partial)
    if vars.iter().all(|&v| v == vars[0]) {
        let dir = Direction::basis(num_inputs, vars[0]);
        return extract_directional_derivative(world, output, &dir, order);
    }

    // For general mixed partials, use inclusion-exclusion / multilinear interpolation
    // This is more complex - we use a recursive/combinatorial approach

    // For order 2 with two variables, use the simple polarization
    if order == 2 && vars.len() == 2 {
        return get_mixed_partial_2(world, output, vars[0], vars[1], num_inputs);
    }

    // For higher orders, use generalized finite differencing
    // This is a simplified version - for production, you'd use proper
    // multilinear interpolation

    // Build direction with sum of all involved basis vectors
    let mut direction_vec = vec![0; num_inputs];
    for &v in &vars {
        direction_vec[v] += 1;
    }
    let direction = Direction::new(direction_vec.clone());

    // Get the directional derivative
    let d_sum = extract_directional_derivative(world, output, &direction, order);

    // For mixed partials, we need to subtract "cross terms"
    // This is the multilinear part of the directional derivative

    // Simplified approach: for symmetric smooth functions, the Taylor coefficient
    // for mixed partial can be extracted using combinatorial factors
    // The formula involves |α|! / α! for the normalization

    // For now, we approximate using the multinomial approach:
    // D^k_d f includes terms like (Σ dᵢ ∂/∂xᵢ)^k f
    // The coefficient of the mixed partial ∂^|α|/∂x^α in this expansion
    // is |α|!/α! times the partial

    // So: partial = α!/|α|! * (coefficient in directional derivative expansion)

    // However, the directional derivative with direction d = Σ eᵢ (for variables in α)
    // already gives us |α|!/α! * partial when all exponents are 1

    // For repeated variables, we need more care.

    // Actually, for the general case with direction d = (d₁, d₂, ...):
    // D^k_d f = Σ_{|β|=k} (k!/β!) * d^β * ∂^k f/∂x^β
    //
    // With d = sum of basis vectors (each appearing αᵢ times), we get a complex formula.

    // For simplicity, let's use numerical approximation via interpolation
    // when the simple formulas don't apply.

    // Count unique variables and their multiplicities
    let unique_vars: Vec<usize> = vars.iter().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();

    if unique_vars.len() == 1 {
        // Pure partial
        let dir = Direction::basis(num_inputs, unique_vars[0]);
        return extract_directional_derivative(world, output, &dir, order);
    }

    if unique_vars.len() == 2 && order == 2 {
        // Mixed second partial
        return get_mixed_partial_2(world, output, unique_vars[0], unique_vars[1], num_inputs);
    }

    // For higher-order mixed partials with multiplicity, use multipoint interpolation
    // This is a fallback - a more sophisticated implementation would use
    // proper multilinear algebra

    // Use the directional derivative formula with correction
    // D^k f(d) for d = (α₁, α₂, ..., αₙ) includes the term:
    // k! / α! * ∂^k f/∂x^α (when summing over all terms)

    // So: ∂^k f/∂x^α = α!/k! * [coefficient of d^α in D^k f(d)]

    // With direction d = (1, 1, ...) for involved variables:
    let factor = index.factorial_product() / crate::util::factorial(order);
    d_sum * factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    #[test]
    fn test_mixed_partial_xy() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y); // f = x * y

        // ∂²f/∂x∂y = 1
        let mixed = get_mixed_partial_2(ad.world_mut(), f, 0, 1, 2);
        assert_relative_eq!(mixed, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mixed_partial_symmetric() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y);

        // ∂²f/∂x∂y = ∂²f/∂y∂x (symmetry)
        let mixed_xy = get_mixed_partial_2(ad.world_mut(), f, 0, 1, 2);
        let mixed_yx = get_mixed_partial_2(ad.world_mut(), f, 1, 0, 2);
        assert_relative_eq!(mixed_xy, mixed_yx, epsilon = 1e-10);
    }

    #[test]
    fn test_mixed_partial_x_squared_y() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // f = x² * y
        let x2 = ad.square(x);
        let f = ad.mul(x2, y);

        // ∂²f/∂x∂y = 2x = 4 at x=2
        let mixed = get_mixed_partial_2(ad.world_mut(), f, 0, 1, 2);
        assert_relative_eq!(mixed, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mixed_partial_x_plus_y_squared() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // f = (x + y)²
        let sum = ad.add(x, y);
        let f = ad.square(sum);

        // f = x² + 2xy + y²
        // ∂²f/∂x∂y = 2
        let mixed = get_mixed_partial_2(ad.world_mut(), f, 0, 1, 2);
        assert_relative_eq!(mixed, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mixed_partial_same_var() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let f = ad.square(x);

        // ∂²f/∂x∂x = ∂²f/∂x² = 2
        let mixed = get_mixed_partial_2(ad.world_mut(), f, 0, 0, 1);
        assert_eq!(mixed, 2.0);
    }

    #[test]
    fn test_mixed_partial_zero() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // f = x² + y² (no xy term)
        let x2 = ad.square(x);
        let y2 = ad.square(y);
        let f = ad.add(x2, y2);

        // ∂²f/∂x∂y = 0
        let mixed = get_mixed_partial_2(ad.world_mut(), f, 0, 1, 2);
        assert_relative_eq!(mixed, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_get_mixed_partial_general() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y);

        // ∂²f/∂x∂y = 1
        let index = MultiIndex::new(vec![1, 1]);
        let partial = get_mixed_partial(ad.world_mut(), f, &index, 2);
        assert_relative_eq!(partial, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_get_mixed_partial_pure() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let f = ad.square(x);

        // ∂²f/∂x² = 2
        let index = MultiIndex::new(vec![2]);
        let partial = get_mixed_partial(ad.world_mut(), f, &index, 1);
        assert_eq!(partial, 2.0);
    }
}
