//! Nested Clenshaw-as-graph for 2D Chebyshev segments.
//!
//! Builds the tensor product Chebyshev evaluation as bevy_autodiff graph nodes.
//! For each x-mode i, an inner Clenshaw in y produces a `Var` result r_i.
//! An outer Clenshaw in x then uses those Var results as "coefficients".
//! Differentiation through the entire 2D evaluation is automatic via the chain rule.

use bevy_autodiff::{AutoDiff, Float, Var};

use crate::fit2d::ChebyshevSegment2D;

impl<F: Float> ChebyshevSegment2D<F> {
    /// Build the nested Clenshaw recurrence as bevy_autodiff graph nodes.
    ///
    /// Given `Var`s for x and y in physical coordinates, constructs graph nodes
    /// for the domain mappings + nested Clenshaw evaluation. Differentiation
    /// through the entire 2D evaluation is automatic via the chain rule.
    ///
    /// Graph size: ~3·N_x·N_y + 3·N_x + 3·N_y + 10 nodes.
    pub fn build_graph(&self, ad: &mut AutoDiff<F>, x: Var, y: Var) -> Var {
        let two = F::from_f64(2.0);

        // Domain mapping: t_x = (2x - (a_x+b_x)) / (b_x-a_x)
        let t_x = build_domain_map(ad, x, self.a_x, self.b_x);
        let t_y = build_domain_map(ad, y, self.a_y, self.b_y);

        // Inner Clenshaw: for each x-mode i, evaluate Σ_j c_ij T_j(t_y)
        let r: Vec<Var> = (0..self.n_x())
            .map(|i| {
                let row_start = i * self.n_y();
                let row = &self.coeffs[row_start..row_start + self.n_y()];
                clenshaw_const_coeffs(ad, row, t_y)
            })
            .collect();

        // Outer Clenshaw: evaluate in x using r_i as Var coefficients
        // The inner Clenshaw handles the y-halving convention (c_{i0}/2).
        // The outer Clenshaw applies x-halving to r_0, giving r_0/2.
        // This correctly produces the double-halved constant term c_{00}/4.
        if r.len() == 1 {
            // Only one x-mode: result is r_0/2
            let half = ad.constant(F::from_f64(0.5));
            return ad.mul(half, r[0]);
        }

        let two_const = ad.constant(two);
        let two_t_x = ad.mul(two_const, t_x);

        let n = r.len() - 1;
        let mut b_next = ad.constant(F::zero());
        let mut b_next2 = ad.constant(F::zero());

        for k in (1..=n).rev() {
            let two_t_b = ad.mul(two_t_x, b_next);
            let sub_term = ad.sub(two_t_b, b_next2);
            let b_curr = ad.add(sub_term, r[k]);
            b_next2 = b_next;
            b_next = b_curr;
        }

        // result = t_x·b_1 - b_2 + r_0/2
        let half = ad.constant(F::from_f64(0.5));
        let r0_half = ad.mul(half, r[0]);
        let t_b1 = ad.mul(t_x, b_next);
        let sub_b2 = ad.sub(t_b1, b_next2);
        ad.add(sub_b2, r0_half)
    }
}

/// Build domain mapping: t = (2x - (a+b)) / (b-a), mapping [a,b] → [-1,1].
fn build_domain_map<F: Float>(ad: &mut AutoDiff<F>, x: Var, a: F, b: F) -> Var {
    let two = F::from_f64(2.0);
    let a_plus_b = ad.constant(a + b);
    let b_minus_a = ad.constant(b - a);
    let two_const = ad.constant(two);
    let two_x = ad.mul(two_const, x);
    let numer = ad.sub(two_x, a_plus_b);
    ad.div(numer, b_minus_a)
}

/// Standard Clenshaw recurrence with constant coefficients, built as graph nodes.
///
/// Evaluates c_0/2 + Σ_{k≥1} c_k T_k(t) where coefficients are `F` values.
fn clenshaw_const_coeffs<F: Float>(ad: &mut AutoDiff<F>, coeffs: &[F], t: Var) -> Var {
    let two = F::from_f64(2.0);

    if coeffs.is_empty() {
        return ad.constant(F::zero());
    }
    if coeffs.len() == 1 {
        return ad.constant(coeffs[0] / two);
    }

    let two_const = ad.constant(two);
    let two_t = ad.mul(two_const, t);

    let n = coeffs.len() - 1;
    let mut b_next = ad.constant(F::zero());
    let mut b_next2 = ad.constant(F::zero());

    for k in (1..=n).rev() {
        let c_k = ad.constant(coeffs[k]);
        let two_t_b = ad.mul(two_t, b_next);
        let sub_term = ad.sub(two_t_b, b_next2);
        let b_curr = ad.add(sub_term, c_k);
        b_next2 = b_next;
        b_next = b_curr;
    }

    let c0_half = ad.constant(coeffs[0] / two);
    let t_b1 = ad.mul(t, b_next);
    let sub_b2 = ad.sub(t_b1, b_next2);
    ad.add(sub_b2, c0_half)
}

#[cfg(test)]
mod tests {
    use crate::fit2d::{FitOptions2D, fit_dense_2d, fit_sparse_2d};
    use bevy_autodiff::AutoDiff;

    #[test]
    fn graph_2d_linear_xy() {
        // f(x,y) = x + y via sparse fit on [0,1]×[0,1]
        // Exact polynomial → Tier 2 tolerance
        let n = 10;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        let mut zd = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                xd.push(x);
                yd.push(y);
                zd.push(x + y);
            }
        }

        let result = fit_sparse_2d(
            &xd,
            &yd,
            &zd,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 2,
                degree_y: 2,
            },
        )
        .unwrap();

        let seg = result.fit.segment(0, 0);
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(0.3).unwrap();
        let y = ad.var(0.7).unwrap();
        let f = seg.build_graph(&mut ad, x, y);

        // f(0.3, 0.7) = 1.0
        let val = ad.eval(f).unwrap();
        assert!(
            (val - 1.0).abs() < 1e-12,
            "f(0.3, 0.7) = {val}, expected 1.0, err = {:.2e}",
            (val - 1.0).abs()
        );

        // ∂f/∂x = 1
        let dfdx = ad.differentiate(f, x).unwrap();
        let dx_val = ad.eval(dfdx).unwrap();
        assert!(
            (dx_val - 1.0).abs() < 1e-12,
            "∂f/∂x = {dx_val}, expected 1.0, err = {:.2e}",
            (dx_val - 1.0).abs()
        );

        // ∂f/∂y = 1
        let dfdy = ad.differentiate(f, y).unwrap();
        let dy_val = ad.eval(dfdy).unwrap();
        assert!(
            (dy_val - 1.0).abs() < 1e-12,
            "∂f/∂y = {dy_val}, expected 1.0, err = {:.2e}",
            (dy_val - 1.0).abs()
        );
    }

    #[test]
    fn graph_2d_x_times_y() {
        // f(x,y) = x*y via sparse fit on [0,1]×[0,1]
        // ∂f/∂x = y, ∂f/∂y = x, ∂²f/∂x∂y = 1
        let n = 10;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        let mut zd = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                xd.push(x);
                yd.push(y);
                zd.push(x * y);
            }
        }

        let result = fit_sparse_2d(
            &xd,
            &yd,
            &zd,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 2,
                degree_y: 2,
            },
        )
        .unwrap();

        let seg = result.fit.segment(0, 0);
        let test_x = 0.4;
        let test_y = 0.6;
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(test_x).unwrap();
        let y = ad.var(test_y).unwrap();
        let f = seg.build_graph(&mut ad, x, y);

        // f(0.4, 0.6) = 0.24
        let val = ad.eval(f).unwrap();
        assert!(
            (val - test_x * test_y).abs() < 1e-12,
            "f({test_x}, {test_y}) = {val}, expected {}, err = {:.2e}",
            test_x * test_y,
            (val - test_x * test_y).abs()
        );

        // ∂f/∂x = y
        let dfdx = ad.differentiate(f, x).unwrap();
        let dx_val = ad.eval(dfdx).unwrap();
        assert!(
            (dx_val - test_y).abs() < 1e-11,
            "∂f/∂x = {dx_val}, expected {test_y}, err = {:.2e}",
            (dx_val - test_y).abs()
        );

        // ∂f/∂y = x
        let dfdy = ad.differentiate(f, y).unwrap();
        let dy_val = ad.eval(dfdy).unwrap();
        assert!(
            (dy_val - test_x).abs() < 1e-11,
            "∂f/∂y = {dy_val}, expected {test_x}, err = {:.2e}",
            (dy_val - test_x).abs()
        );

        // ∂²f/∂x∂y = 1
        let d2fdxdy = ad.differentiate(dfdx, y).unwrap();
        let dxy_val = ad.eval(d2fdxdy).unwrap();
        assert!(
            (dxy_val - 1.0).abs() < 1e-10,
            "∂²f/∂x∂y = {dxy_val}, expected 1.0, err = {:.2e}",
            (dxy_val - 1.0).abs()
        );
    }

    #[test]
    fn graph_2d_sin_cos_derivative() {
        // f(x,y) = sin(x)*cos(y) via dense fit on [0,π]×[0,π]
        // ∂f/∂x = cos(x)*cos(y)
        let pi = std::f64::consts::PI;
        let nx = 100;
        let ny = 100;
        let x_data: Vec<f64> = (0..=nx).map(|i| pi * i as f64 / nx as f64).collect();
        let y_data: Vec<f64> = (0..=ny).map(|j| pi * j as f64 / ny as f64).collect();
        let z_data: Vec<Vec<f64>> = y_data
            .iter()
            .map(|&y| x_data.iter().map(|&x| x.sin() * y.cos()).collect())
            .collect();

        let result = fit_dense_2d(
            &x_data,
            &y_data,
            &z_data,
            &[0.0, pi],
            &[0.0, pi],
            &FitOptions2D {
                degree_x: 16,
                degree_y: 16,
            },
        )
        .unwrap();

        let seg = result.fit.segment(0, 0);
        let test_x = 1.0;
        let test_y = 0.5;
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(test_x).unwrap();
        let y = ad.var(test_y).unwrap();
        let f = seg.build_graph(&mut ad, x, y);

        // Check value (Tier 3)
        let val = ad.eval(f).unwrap();
        let expected = test_x.sin() * test_y.cos();
        assert!(
            (val - expected).abs() < 5e-4,
            "f({test_x}, {test_y}) = {val}, expected {expected}, err = {:.2e}",
            (val - expected).abs()
        );

        // Check ∂f/∂x ≈ cos(x)*cos(y) (Tier 3, derivative amplifies error)
        let dfdx = ad.differentiate(f, x).unwrap();
        let dx_val = ad.eval(dfdx).unwrap();
        let expected_dx = test_x.cos() * test_y.cos();
        assert!(
            (dx_val - expected_dx).abs() < 5e-3,
            "∂f/∂x = {dx_val}, expected {expected_dx}, err = {:.2e}",
            (dx_val - expected_dx).abs()
        );
    }

    #[test]
    fn graph_2d_compose_with_exp() {
        // g(x,y) = exp(f(x,y)) where f = x*y via sparse fit
        // dg/dx = exp(f) * df/dx = exp(x*y) * y
        let n = 10;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        let mut zd = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                xd.push(x);
                yd.push(y);
                zd.push(x * y);
            }
        }

        let result = fit_sparse_2d(
            &xd,
            &yd,
            &zd,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 2,
                degree_y: 2,
            },
        )
        .unwrap();

        let seg = result.fit.segment(0, 0);
        let test_x = 0.5;
        let test_y = 0.4;
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(test_x).unwrap();
        let y = ad.var(test_y).unwrap();
        let f = seg.build_graph(&mut ad, x, y);
        let g = ad.exp(f);

        // g(0.5, 0.4) = exp(0.2)
        let val = ad.eval(g).unwrap();
        let expected = (test_x * test_y).exp();
        assert!(
            (val - expected).abs() < 1e-10,
            "g({test_x}, {test_y}) = {val}, expected {expected}, err = {:.2e}",
            (val - expected).abs()
        );

        // dg/dx = exp(x*y) * y
        let dgdx = ad.differentiate(g, x).unwrap();
        let dx_val = ad.eval(dgdx).unwrap();
        let expected_dx = expected * test_y;
        assert!(
            (dx_val - expected_dx).abs() < 1e-9,
            "dg/dx = {dx_val}, expected {expected_dx}, err = {:.2e}",
            (dx_val - expected_dx).abs()
        );
    }

    #[test]
    fn graph_2d_matches_eval() {
        // Graph evaluation should match standalone eval (Tier 2)
        let n = 10;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        let mut zd = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                xd.push(x);
                yd.push(y);
                zd.push(x * x + 2.0 * x * y + 3.0 * y * y);
            }
        }

        let result = fit_sparse_2d(
            &xd,
            &yd,
            &zd,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 3,
                degree_y: 3,
            },
        )
        .unwrap();

        let seg = result.fit.segment(0, 0);

        for &(tx, ty) in &[(0.1, 0.2), (0.5, 0.5), (0.8, 0.3), (0.0, 1.0)] {
            let standalone = seg.eval(tx, ty);
            let mut ad = AutoDiff::<f64>::new();
            let x = ad.var(tx).unwrap();
            let y = ad.var(ty).unwrap();
            let f = seg.build_graph(&mut ad, x, y);
            let graph_val = ad.eval(f).unwrap();
            assert!(
                (graph_val - standalone).abs() < 1e-12,
                "at ({tx}, {ty}): graph={graph_val}, standalone={standalone}, err={:.2e}",
                (graph_val - standalone).abs()
            );
        }
    }
}
