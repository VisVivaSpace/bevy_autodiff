//! Reverse mode gradient computation.
//!
//! This module provides the main entry point for computing gradients via
//! reverse mode (backpropagation) through the computation graph.

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use std::collections::HashMap;

use crate::components::{BinaryInputs, BinaryOp, Direction, IsConstant, IsInput, UnaryInput, UnaryOp};
use crate::context::{BinaryOpMarker, UnaryOpMarker};
use crate::graph::topology::topological_order;
use crate::taylor::propagate::propagate_taylor;
use crate::var::Var;

use super::adjoint_rules::{
    adjoint_acos, adjoint_acosh, adjoint_add, adjoint_asin, adjoint_asinh, adjoint_atan,
    adjoint_atanh, adjoint_div, adjoint_exp, adjoint_ln, adjoint_mul, adjoint_neg, adjoint_sqrt,
    adjoint_sub, adjoint_tan, adjoint_tanh,
};

/// Accumulates adjoint contributions from an output back to all inputs.
///
/// This is the main reverse mode function that:
/// 1. Runs a forward pass to compute Taylor coefficients
/// 2. Initializes the output adjoint to 1
/// 3. Propagates adjoints backward through the graph in reverse topological order
/// 4. Returns the adjoint of each input (which is the gradient)
///
/// # Arguments
/// - `world`: The ECS world
/// - `output`: The output variable to differentiate
/// - `direction`: The direction for Taylor expansion
/// - `order`: The order of derivatives to compute
///
/// # Returns
/// A map from input entity to its adjoint Taylor coefficients
pub fn reverse_accumulate(
    world: &mut World,
    output: Entity,
    direction: &Direction,
    order: usize,
) -> HashMap<Entity, Vec<f64>> {
    // Step 1: Forward pass - compute Taylor coefficients for all nodes
    let topo_order = topological_order(world, output);

    // Propagate Taylor coefficients in forward order
    let mut taylor_cache: HashMap<Entity, Vec<f64>> = HashMap::new();
    for &entity in &topo_order {
        let coeffs = propagate_taylor(world, entity, direction, order);
        taylor_cache.insert(entity, coeffs);
    }

    // Step 2: Initialize adjoints
    // Output adjoint is [1, 0, 0, ...] (we want df/d(output) = 1)
    let mut adjoint_cache: HashMap<Entity, Vec<f64>> = HashMap::new();
    let output_adj = {
        let mut adj = vec![0.0; order + 1];
        adj[0] = 1.0;
        adj
    };
    adjoint_cache.insert(output, output_adj);

    // Step 3: Backward pass - propagate adjoints in reverse topological order
    for &entity in topo_order.iter().rev() {
        // Get this node's adjoint (skip if not computed yet - could be disconnected)
        let entity_adj = match adjoint_cache.get(&entity) {
            Some(adj) => adj.clone(),
            None => continue,
        };

        // Get this node's Taylor coefficients
        let entity_taylor = match taylor_cache.get(&entity) {
            Some(t) => t.clone(),
            None => continue,
        };

        // Check what type of node this is and propagate accordingly
        let entity_ref = world.entity(entity);

        // Skip inputs and constants - they are leaves
        if entity_ref.contains::<IsInput>() || entity_ref.contains::<IsConstant>() {
            continue;
        }

        // Handle unary operations
        if let Some(unary_marker) = entity_ref.get::<UnaryOpMarker>() {
            let op = unary_marker.0;
            if let Some(unary_input) = entity_ref.get::<UnaryInput>() {
                let input_entity = unary_input.get().entity();
                let input_taylor = taylor_cache.get(&input_entity).cloned().unwrap_or_default();

                // Get or create input adjoint
                let input_adj = adjoint_cache
                    .entry(input_entity)
                    .or_insert_with(|| vec![0.0; order + 1]);

                // Apply adjoint rule based on operation
                match op {
                    UnaryOp::Neg => {
                        adjoint_neg(&entity_adj, input_adj);
                    }
                    UnaryOp::Exp => {
                        adjoint_exp(&entity_adj, &entity_taylor, input_adj);
                    }
                    UnaryOp::Ln => {
                        adjoint_ln(&entity_adj, &input_taylor, input_adj);
                    }
                    UnaryOp::Sqrt => {
                        adjoint_sqrt(&entity_adj, &entity_taylor, input_adj);
                    }
                    UnaryOp::Sin | UnaryOp::Cos => {
                        // Sin and cos need special handling - they share coupled computations
                        // For now, we compute the derivative directly
                        // sin'(u) = cos(u), cos'(u) = -sin(u)
                        let cos_taylor = if op == UnaryOp::Sin {
                            // Need to compute cos of the same input
                            use crate::taylor::rules::coupled::{coupled_taylor, SinCos};
                            let outputs = coupled_taylor::<SinCos>(&input_taylor, order);
                            outputs[1].clone() // cos is second output
                        } else {
                            entity_taylor.clone()
                        };

                        let sin_taylor = if op == UnaryOp::Cos {
                            use crate::taylor::rules::coupled::{coupled_taylor, SinCos};
                            let outputs = coupled_taylor::<SinCos>(&input_taylor, order);
                            outputs[0].clone() // sin is first output
                        } else {
                            entity_taylor.clone()
                        };

                        if op == UnaryOp::Sin {
                            // Only sin adjoint, cos adjoint is 0
                            let zero_adj = vec![0.0; order + 1];
                            super::adjoint_rules::adjoint_sin_cos(
                                &entity_adj,
                                &zero_adj,
                                &sin_taylor,
                                &cos_taylor,
                                input_adj,
                            );
                        } else {
                            // Only cos adjoint, sin adjoint is 0
                            let zero_adj = vec![0.0; order + 1];
                            super::adjoint_rules::adjoint_sin_cos(
                                &zero_adj,
                                &entity_adj,
                                &sin_taylor,
                                &cos_taylor,
                                input_adj,
                            );
                        }
                    }
                    UnaryOp::Sinh | UnaryOp::Cosh => {
                        let cosh_taylor = if op == UnaryOp::Sinh {
                            use crate::taylor::rules::coupled::{coupled_taylor, SinhCosh};
                            let outputs = coupled_taylor::<SinhCosh>(&input_taylor, order);
                            outputs[1].clone()
                        } else {
                            entity_taylor.clone()
                        };

                        let sinh_taylor = if op == UnaryOp::Cosh {
                            use crate::taylor::rules::coupled::{coupled_taylor, SinhCosh};
                            let outputs = coupled_taylor::<SinhCosh>(&input_taylor, order);
                            outputs[0].clone()
                        } else {
                            entity_taylor.clone()
                        };

                        if op == UnaryOp::Sinh {
                            let zero_adj = vec![0.0; order + 1];
                            super::adjoint_rules::adjoint_sinh_cosh(
                                &entity_adj,
                                &zero_adj,
                                &sinh_taylor,
                                &cosh_taylor,
                                input_adj,
                            );
                        } else {
                            let zero_adj = vec![0.0; order + 1];
                            super::adjoint_rules::adjoint_sinh_cosh(
                                &zero_adj,
                                &entity_adj,
                                &sinh_taylor,
                                &cosh_taylor,
                                input_adj,
                            );
                        }
                    }
                    // tan and tanh: use specialized adjoint rules
                    UnaryOp::Tan => {
                        adjoint_tan(&entity_adj, &entity_taylor, input_adj);
                    }
                    UnaryOp::Tanh => {
                        adjoint_tanh(&entity_adj, &entity_taylor, input_adj);
                    }
                    // Inverse trig functions: use specialized adjoint rules
                    UnaryOp::Asin => {
                        adjoint_asin(&entity_adj, &input_taylor, input_adj);
                    }
                    UnaryOp::Acos => {
                        adjoint_acos(&entity_adj, &input_taylor, input_adj);
                    }
                    UnaryOp::Atan => {
                        adjoint_atan(&entity_adj, &input_taylor, input_adj);
                    }
                    // Inverse hyperbolic functions: use specialized adjoint rules
                    UnaryOp::Asinh => {
                        adjoint_asinh(&entity_adj, &input_taylor, input_adj);
                    }
                    UnaryOp::Acosh => {
                        adjoint_acosh(&entity_adj, &input_taylor, input_adj);
                    }
                    UnaryOp::Atanh => {
                        adjoint_atanh(&entity_adj, &input_taylor, input_adj);
                    }
                }
            }
        }

        // Handle binary operations
        if let Some(binary_marker) = entity_ref.get::<BinaryOpMarker>() {
            let op = binary_marker.0;
            if let Some(binary_inputs) = entity_ref.get::<BinaryInputs>() {
                let left_entity = binary_inputs.left.entity();
                let right_entity = binary_inputs.right.entity();

                let left_taylor = taylor_cache.get(&left_entity).cloned().unwrap_or_default();
                let right_taylor = taylor_cache.get(&right_entity).cloned().unwrap_or_default();

                // Check if left and right are the same entity (e.g., x * x)
                let same_entity = left_entity == right_entity;

                // Compute adjoint contributions
                let mut left_contrib = vec![0.0; order + 1];
                let mut right_contrib = vec![0.0; order + 1];

                // Apply adjoint rule based on operation
                match op {
                    BinaryOp::Add => {
                        adjoint_add(&entity_adj, &mut left_contrib, &mut right_contrib);
                    }
                    BinaryOp::Sub => {
                        adjoint_sub(&entity_adj, &mut left_contrib, &mut right_contrib);
                    }
                    BinaryOp::Mul => {
                        adjoint_mul(
                            &entity_adj,
                            &left_taylor,
                            &right_taylor,
                            &mut left_contrib,
                            &mut right_contrib,
                        );
                    }
                    BinaryOp::Div => {
                        adjoint_div(
                            &entity_adj,
                            &left_taylor,
                            &right_taylor,
                            &entity_taylor,
                            &mut left_contrib,
                            &mut right_contrib,
                        );
                    }
                    BinaryOp::Pow => {
                        // y = u^v = exp(v * ln(u))
                        // Check if right (exponent) is constant
                        let right_is_const = world.entity(right_entity).contains::<IsConstant>();

                        if right_is_const {
                            let p = right_taylor.first().copied().unwrap_or(0.0);
                            super::adjoint_rules::adjoint_pow_const(
                                &entity_adj,
                                &left_taylor,
                                &entity_taylor,
                                p,
                                &mut left_contrib,
                            );
                        } else {
                            // General case: y = u^v = exp(v * ln(u))
                            // dy/du = v * u^(v-1) = v * y / u
                            // dy/dv = u^v * ln(u) = y * ln(u)

                            let adj_order = entity_adj.len().saturating_sub(1);
                            let y_over_u = crate::taylor::polynomial::div_taylor(
                                &entity_taylor,
                                &left_taylor,
                                adj_order,
                            ).expect("division by zero in pow adjoint");
                            let v_y_over_u =
                                crate::taylor::polynomial::mul_taylor(&right_taylor, &y_over_u, adj_order);
                            let left_c =
                                crate::taylor::polynomial::mul_taylor(&v_y_over_u, &entity_adj, adj_order);
                            for (i, &c) in left_c.iter().enumerate() {
                                if i < left_contrib.len() {
                                    left_contrib[i] += c;
                                }
                            }

                            let ln_u = crate::taylor::rules::elementary::ln_taylor(&left_taylor, adj_order).expect("ln domain error in pow adjoint");
                            let y_ln_u =
                                crate::taylor::polynomial::mul_taylor(&entity_taylor, &ln_u, adj_order);
                            let right_c =
                                crate::taylor::polynomial::mul_taylor(&y_ln_u, &entity_adj, adj_order);
                            for (i, &c) in right_c.iter().enumerate() {
                                if i < right_contrib.len() {
                                    right_contrib[i] += c;
                                }
                            }
                        }
                    }
                }

                // Accumulate contributions to adjoint cache
                if same_entity {
                    // Both contributions go to the same entity - add them together
                    let adj = adjoint_cache
                        .entry(left_entity)
                        .or_insert_with(|| vec![0.0; order + 1]);
                    for (i, (&l, &r)) in left_contrib.iter().zip(right_contrib.iter()).enumerate() {
                        if i < adj.len() {
                            adj[i] += l + r;
                        }
                    }
                } else {
                    // Different entities - accumulate separately
                    let left_adj = adjoint_cache
                        .entry(left_entity)
                        .or_insert_with(|| vec![0.0; order + 1]);
                    for (i, &c) in left_contrib.iter().enumerate() {
                        if i < left_adj.len() {
                            left_adj[i] += c;
                        }
                    }

                    let right_adj = adjoint_cache
                        .entry(right_entity)
                        .or_insert_with(|| vec![0.0; order + 1]);
                    for (i, &c) in right_contrib.iter().enumerate() {
                        if i < right_adj.len() {
                            right_adj[i] += c;
                        }
                    }
                }
            }
        }
    }

    // Return only input adjoints
    adjoint_cache
        .into_iter()
        .filter(|(entity, _)| world.entity(*entity).contains::<IsInput>())
        .collect()
}

/// Computes the gradient of a scalar output with respect to all inputs using reverse mode.
///
/// This is more efficient than forward mode when there are many inputs and one output.
///
/// # Arguments
/// - `world`: The ECS world
/// - `output`: The output variable (should be a scalar)
/// - `num_inputs`: The total number of input variables
///
/// # Returns
/// A vector of gradients, one for each input in order of creation
pub fn compute_gradient_reverse(world: &mut World, output: Var, num_inputs: usize) -> Vec<f64> {
    let direction = Direction::zero(num_inputs);

    // Run reverse accumulation
    let adjoints = reverse_accumulate(world, output.entity(), &direction, 0);

    // Extract first-order derivatives (gradient)
    // We need to map entities back to their input indices
    let mut gradient = vec![0.0; num_inputs];

    for (entity, adj) in adjoints {
        // Get the input index from Dependencies
        if let Some(deps) = world.entity(entity).get::<crate::components::Dependencies>() {
            if !deps.is_empty() {
                let index = deps.mask.trailing_zeros() as usize;
                if index < num_inputs && !adj.is_empty() {
                    gradient[index] = adj[0];
                }
            }
        }
    }

    gradient
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    #[test]
    fn test_reverse_gradient_x_squared() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let y = ad.square(x); // y = x²

        // dy/dx = 2x = 6
        let grad = compute_gradient_reverse(ad.world_mut(), y, 1);
        assert_eq!(grad.len(), 1);
        assert_relative_eq!(grad[0], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_xy() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y); // f = xy

        // df/dx = y = 3, df/dy = x = 2
        let grad = compute_gradient_reverse(ad.world_mut(), f, 2);
        assert_eq!(grad.len(), 2);
        assert_relative_eq!(grad[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(grad[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_sum_of_squares() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(2.0);
        let x2 = ad.square(x);
        let y2 = ad.square(y);
        let f = ad.add(x2, y2); // f = x² + y²

        // df/dx = 2x = 2, df/dy = 2y = 4
        let grad = compute_gradient_reverse(ad.world_mut(), f, 2);
        assert_eq!(grad.len(), 2);
        assert_relative_eq!(grad[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(grad[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_matches_forward() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // f = x² * y + x * y²
        let x2 = ad.square(x);
        let y2 = ad.square(y);
        let x2y = ad.mul(x2, y);
        let xy2 = ad.mul(x, y2);
        let f = ad.add(x2y, xy2);

        // Forward mode gradient
        let forward_grad = ad.gradient(f);

        // Reverse mode gradient
        let reverse_grad = compute_gradient_reverse(ad.world_mut(), f, 2);

        assert_eq!(forward_grad.len(), reverse_grad.len());
        for (fw, rv) in forward_grad.iter().zip(reverse_grad.iter()) {
            assert_relative_eq!(fw, rv, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reverse_gradient_exp() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.exp(x); // f = e^x

        // df/dx = e^x = e
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], std::f64::consts::E, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_ln() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let f = ad.ln(x); // f = ln(x)

        // df/dx = 1/x = 0.5
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_sqrt() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);
        let f = ad.sqrt(x); // f = sqrt(x)

        // df/dx = 1/(2*sqrt(x)) = 1/4 = 0.25
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_chain() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let sin_x = ad.sin(x);
        let f = ad.exp(sin_x); // f = exp(sin(x))

        // df/dx = exp(sin(x)) * cos(x) = 1 * 1 = 1 at x=0
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_division() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(4.0);
        let f = ad.div(x, y); // f = x/y

        // df/dx = 1/y = 0.25
        // df/dy = -x/y² = -2/16 = -0.125
        let grad = compute_gradient_reverse(ad.world_mut(), f, 2);
        assert_relative_eq!(grad[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(grad[1], -0.125, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_subtraction() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let y = ad.var(3.0);
        let f = ad.sub(x, y); // f = x - y

        // df/dx = 1, df/dy = -1
        let grad = compute_gradient_reverse(ad.world_mut(), f, 2);
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(grad[1], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_negation() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let f = ad.neg(x); // f = -x

        // df/dx = -1
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_power() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let f = ad.powi(x, 3); // f = x³

        // df/dx = 3x² = 12
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rosenbrock_gradient() {
        // Rosenbrock function: f(x, y) = (a - x)² + b(y - x²)²
        // with a=1, b=100
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let y = ad.var(0.0);

        let a = ad.constant(1.0);
        let b = ad.constant(100.0);

        // (a - x)²
        let a_minus_x = ad.sub(a, x);
        let term1 = ad.square(a_minus_x);

        // b(y - x²)²
        let x2 = ad.square(x);
        let y_minus_x2 = ad.sub(y, x2);
        let y_minus_x2_sq = ad.square(y_minus_x2);
        let term2 = ad.mul(b, y_minus_x2_sq);

        let f = ad.add(term1, term2);

        // At (0, 0):
        // df/dx = -2(1-x) + 100*2(y-x²)*(-2x) = -2 + 0 = -2
        // df/dy = 100*2(y-x²) = 0
        let grad = compute_gradient_reverse(ad.world_mut(), f, 2);
        assert_relative_eq!(grad[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(grad[1], 0.0, epsilon = 1e-10);
    }

    // =========================================================================
    // Tests for new specialized adjoint rules (tan, tanh, inverse trig/hyperbolic)
    // =========================================================================

    #[test]
    fn test_reverse_gradient_tan() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let f = ad.tan(x); // f = tan(x)

        // df/dx = sec²(x) = 1 at x=0
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_tanh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let f = ad.tanh(x); // f = tanh(x)

        // df/dx = sech²(x) = 1 at x=0
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_asin() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let f = ad.asin(x); // f = asin(x)

        // df/dx = 1/√(1-x²) = 1 at x=0
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_acos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let f = ad.acos(x); // f = acos(x)

        // df/dx = -1/√(1-x²) = -1 at x=0
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_atan() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let f = ad.atan(x); // f = atan(x)

        // df/dx = 1/(1+x²) = 0.5 at x=1
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_asinh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);
        let f = ad.asinh(x); // f = asinh(x)

        // df/dx = 1/√(x²+1) = 1 at x=0
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_acosh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let f = ad.acosh(x); // f = acosh(x)

        // df/dx = 1/√(x²-1) = 1/√3 at x=2
        let expected = 1.0 / (3.0_f64).sqrt();
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_atanh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.atanh(x); // f = atanh(x)

        // df/dx = 1/(1-x²) = 1/(1-0.25) = 4/3 at x=0.5
        let expected = 1.0 / 0.75;
        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_matches_forward_tan() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);
        let f = ad.tan(x);

        let forward_grad = ad.gradient(f);
        let reverse_grad = compute_gradient_reverse(ad.world_mut(), f, 1);

        assert_relative_eq!(forward_grad[0], reverse_grad[0], epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_matches_forward_inverse_trig() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);

        // Test asin
        let f_asin = ad.asin(x);
        let fwd_asin = ad.gradient(f_asin);
        let rev_asin = compute_gradient_reverse(ad.world_mut(), f_asin, 1);
        assert_relative_eq!(fwd_asin[0], rev_asin[0], epsilon = 1e-10);

        // Test atan
        let f_atan = ad.atan(x);
        let fwd_atan = ad.gradient(f_atan);
        let rev_atan = compute_gradient_reverse(ad.world_mut(), f_atan, 1);
        assert_relative_eq!(fwd_atan[0], rev_atan[0], epsilon = 1e-10);

        // Test asinh
        let f_asinh = ad.asinh(x);
        let fwd_asinh = ad.gradient(f_asinh);
        let rev_asinh = compute_gradient_reverse(ad.world_mut(), f_asinh, 1);
        assert_relative_eq!(fwd_asinh[0], rev_asinh[0], epsilon = 1e-10);

        // Test atanh
        let f_atanh = ad.atanh(x);
        let fwd_atanh = ad.gradient(f_atanh);
        let rev_atanh = compute_gradient_reverse(ad.world_mut(), f_atanh, 1);
        assert_relative_eq!(fwd_atanh[0], rev_atanh[0], epsilon = 1e-10);
    }

    #[test]
    fn test_reverse_gradient_composition_with_tan() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5);

        // f = tan(x²)
        let x2 = ad.square(x);
        let f = ad.tan(x2);

        // df/dx = sec²(x²) · 2x
        // At x=0.5: sec²(0.25) · 1 = (1 + tan²(0.25)) · 1
        let tan_x2 = 0.25_f64.tan();
        let expected = (1.0 + tan_x2 * tan_x2) * 1.0; // 2x = 1 at x=0.5

        let grad = compute_gradient_reverse(ad.world_mut(), f, 1);
        assert_relative_eq!(grad[0], expected, epsilon = 1e-10);
    }
}
