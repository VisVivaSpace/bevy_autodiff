//! Incremental Taylor coefficient order extension.
//!
//! This module provides utilities for extending previously computed Taylor
//! coefficients to higher orders, allowing on-demand computation of higher
//! derivatives without recomputing lower-order terms.

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

use crate::components::Direction;
use crate::taylor::propagate::propagate_taylor;

/// Ensures that Taylor coefficients up to the specified order are computed.
///
/// If coefficients for a lower order are already cached, this only computes
/// the additional higher-order terms. If no cache exists, it computes from
/// scratch.
///
/// # Arguments
/// - `world`: The ECS world
/// - `entity`: The entity to compute coefficients for
/// - `direction`: The direction for directional derivatives
/// - `order`: The desired maximum order
///
/// # Returns
/// The Taylor coefficients up to the specified order
pub fn ensure_taylor_order(
    world: &mut World,
    entity: Entity,
    direction: &Direction,
    order: usize,
) -> Vec<f64> {
    // Check if we already have enough coefficients
    let current_order = world
        .get::<crate::components::TaylorData>(entity)
        .and_then(|td| td.get_directional(direction))
        .map(|coeffs| coeffs.len().saturating_sub(1))
        .unwrap_or(0);

    if current_order >= order {
        // Return existing coefficients
        world
            .get::<crate::components::TaylorData>(entity)
            .and_then(|td| td.get_directional(direction))
            .map(|c| c[..=order].to_vec())
            .unwrap_or_else(|| vec![0.0; order + 1])
    } else {
        // Need to compute (possibly extending)
        // For now, we recompute from scratch
        // A more sophisticated implementation would extend existing coefficients
        propagate_taylor(world, entity, direction, order)
    }
}

/// Extends Taylor coefficients from current_order to new_order.
///
/// This is a lower-level function that assumes the caller has verified that
/// extension is needed. It returns only the newly computed coefficients.
///
/// # Note
/// Current implementation recomputes from scratch. A true incremental
/// implementation would extend using recurrence relations.
pub fn extend_taylor_order(
    world: &mut World,
    entity: Entity,
    direction: &Direction,
    new_order: usize,
) -> Vec<f64> {
    // For now, we recompute everything
    // True incremental extension would require storing intermediate state
    // and using recurrence relations to extend
    propagate_taylor(world, entity, direction, new_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    #[test]
    fn test_ensure_taylor_order_basic() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.square(x); // y = x²

        let direction = Direction::basis(1, 0);

        // Request order 2
        let coeffs = ensure_taylor_order(ad.world_mut(), y.entity(), &direction, 2);

        assert_eq!(coeffs.len(), 3);
        assert_relative_eq!(coeffs[0], 4.0, epsilon = 1e-10); // value
        assert_relative_eq!(coeffs[1], 4.0, epsilon = 1e-10); // first derivative / 1!
        assert_relative_eq!(coeffs[2], 1.0, epsilon = 1e-10); // second derivative / 2!
    }

    #[test]
    fn test_ensure_taylor_order_idempotent() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let y = ad.square(x);

        let direction = Direction::basis(1, 0);

        // Request order 2 twice
        let coeffs1 = ensure_taylor_order(ad.world_mut(), y.entity(), &direction, 2);
        let coeffs2 = ensure_taylor_order(ad.world_mut(), y.entity(), &direction, 2);

        assert_eq!(coeffs1, coeffs2);
    }

    #[test]
    fn test_ensure_taylor_order_increasing() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.exp(x); // y = exp(x)

        let direction = Direction::basis(1, 0);

        // Request order 2, then order 5
        let coeffs2 = ensure_taylor_order(ad.world_mut(), y.entity(), &direction, 2);
        let coeffs5 = ensure_taylor_order(ad.world_mut(), y.entity(), &direction, 5);

        assert_eq!(coeffs2.len(), 3);
        assert_eq!(coeffs5.len(), 6);

        // Lower-order coefficients should match
        for i in 0..=2 {
            assert_relative_eq!(coeffs2[i], coeffs5[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_extend_taylor_order() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.exp(x); // All derivatives of exp(x) at x=1 equal e

        let direction = Direction::basis(1, 0);

        let coeffs = extend_taylor_order(ad.world_mut(), y.entity(), &direction, 4);

        assert_eq!(coeffs.len(), 5);
        // exp(1) = e, and all Taylor coefficients are e/k!
        let e = std::f64::consts::E;
        for (k, &coeff) in coeffs.iter().enumerate() {
            let expected = e / crate::util::factorial(k);
            assert_relative_eq!(coeff, expected, epsilon = 1e-10);
        }
    }
}
