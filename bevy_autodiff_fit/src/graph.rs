//! Clenshaw-as-graph: build Chebyshev evaluation as bevy_autodiff graph nodes.
//!
//! This is the core integration with bevy_autodiff. The Clenshaw recurrence
//! is built as a sequence of `add`, `mul`, `sub`, `constant` operations in the
//! computation graph. Differentiation then works automatically via the chain rule.

use bevy_autodiff::{AutoDiff, Float, Var};

use crate::error::FitError;
use crate::fit::ChebyshevSegment;
use crate::piecewise::PiecewiseFit;

impl<F: Float> ChebyshevSegment<F> {
    /// Build the Clenshaw recurrence as bevy_autodiff graph nodes.
    ///
    /// Given a `Var` for x in the physical domain [a, b], constructs:
    /// 1. Linear mapping: `t = (2·x - (a+b)) / (b-a)` mapping [a,b] → [-1,1]
    /// 2. Clenshaw recurrence: `b_k = 2·t·b_{k+1} - b_{k+2} + c_k`
    /// 3. Final combination: `result = t·b_1 - b_2 + c_0/2`
    ///
    /// Returns a `Var` representing the polynomial value at x.
    /// The chain rule through the mapping + recurrence is automatic —
    /// calling `ad.differentiate(result, x)` gives the exact derivative.
    ///
    /// For a degree-N polynomial, creates ~3N + 5 graph nodes.
    pub fn build_graph(&self, ad: &mut AutoDiff<F>, x: Var) -> Var {
        let coeffs = &self.coeffs;

        if coeffs.is_empty() {
            return ad.constant(F::zero());
        }
        if coeffs.len() == 1 {
            return ad.constant(coeffs[0] / F::from_f64(2.0));
        }

        // Step 1: Linear mapping x ∈ [a, b] → t ∈ [-1, 1]
        // t = (2·x - (a+b)) / (b-a)
        let two = F::from_f64(2.0);
        let a_plus_b = ad.constant(self.a + self.b);
        let b_minus_a = ad.constant(self.b - self.a);
        let two_const = ad.constant(two);
        let two_x = ad.mul(two_const, x);
        let numer = ad.sub(two_x, a_plus_b);
        let t = ad.div(numer, b_minus_a);

        // Step 2: Clenshaw recurrence
        // b_{N+1} = b_{N+2} = 0
        // b_k = 2·t·b_{k+1} - b_{k+2} + c_k   for k = N, N-1, ..., 1
        let two_const2 = ad.constant(two);
        let two_t = ad.mul(two_const2, t);

        let n = coeffs.len() - 1; // degree
        let mut b_next = ad.constant(F::zero()); // b_{k+1}
        let mut b_next2 = ad.constant(F::zero()); // b_{k+2}

        for k in (1..=n).rev() {
            let c_k = ad.constant(coeffs[k]);
            let two_t_b = ad.mul(two_t, b_next);
            let sub_term = ad.sub(two_t_b, b_next2);
            let b_curr = ad.add(sub_term, c_k);
            b_next2 = b_next;
            b_next = b_curr;
        }

        // Step 3: Final combination
        // result = t·b_1 - b_2 + c_0/2
        let c0_half = ad.constant(coeffs[0] / two);
        let t_b1 = ad.mul(t, b_next);
        let sub_b2 = ad.sub(t_b1, b_next2);
        ad.add(sub_b2, c0_half)
    }
}

impl<F: Float> PiecewiseFit<F> {
    /// Build a Clenshaw-as-graph for a specific segment.
    ///
    /// The caller specifies which segment to build the graph for. This is
    /// necessary because bevy_autodiff graphs are static (no branching) —
    /// runtime segment selection happens outside the graph.
    ///
    /// # Errors
    ///
    /// Returns `FitError::SegmentOutOfRange` if the segment index is invalid.
    pub fn build_segment_graph(
        &self,
        ad: &mut AutoDiff<F>,
        x: Var,
        segment: usize,
    ) -> Result<Var, FitError> {
        if segment >= self.num_segments() {
            return Err(FitError::SegmentOutOfRange {
                index: segment,
                count: self.num_segments(),
            });
        }
        Ok(self.segment(segment).build_graph(ad, x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{FitOptions, fit_dense};

    #[test]
    fn graph_constant_polynomial() {
        // f(x) = 5 on [0, 1]
        let seg = ChebyshevSegment {
            coeffs: vec![10.0], // c_0/2 = 5
            a: 0.0,
            b: 1.0,
        };
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(0.5).unwrap();
        let f = seg.build_graph(&mut ad, x);
        assert!((ad.eval(f).unwrap() - 5.0).abs() < 1e-14);

        // Derivative of constant is 0
        let dfdx = ad.differentiate(f, x).unwrap();
        assert!(ad.eval(dfdx).unwrap().abs() < 1e-14);
    }

    #[test]
    fn graph_linear_on_unit_interval() {
        // f(t) = T_1(t) = t on [-1, 1]
        // coeffs: c_0 = 0, c_1 = 1
        let seg = ChebyshevSegment {
            coeffs: vec![0.0, 1.0],
            a: -1.0,
            b: 1.0,
        };
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(0.6).unwrap();
        let f = seg.build_graph(&mut ad, x);

        assert!(
            (ad.eval(f).unwrap() - 0.6).abs() < 1e-13,
            "f(0.6) = {}",
            ad.eval(f).unwrap()
        );

        // f'(x) = 1
        let dfdx = ad.differentiate(f, x).unwrap();
        assert!(
            (ad.eval(dfdx).unwrap() - 1.0).abs() < 1e-13,
            "f'(0.6) = {}",
            ad.eval(dfdx).unwrap()
        );
    }

    #[test]
    fn graph_t2_derivative() {
        // f(t) = T_2(t) = 2t²-1 on [-1, 1]
        // f'(t) = 4t
        let seg = ChebyshevSegment {
            coeffs: vec![0.0, 0.0, 1.0],
            a: -1.0,
            b: 1.0,
        };
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(0.5).unwrap();
        let f = seg.build_graph(&mut ad, x);

        let expected_val = 2.0 * 0.5 * 0.5 - 1.0; // -0.5
        assert!(
            (ad.eval(f).unwrap() - expected_val).abs() < 1e-13,
            "f(0.5) = {}",
            ad.eval(f).unwrap()
        );

        let dfdx = ad.differentiate(f, x).unwrap();
        let expected_deriv = 4.0 * 0.5; // 2.0
        assert!(
            (ad.eval(dfdx).unwrap() - expected_deriv).abs() < 1e-12,
            "f'(0.5) = {}",
            ad.eval(dfdx).unwrap()
        );
    }

    #[test]
    fn graph_mapped_domain() {
        // f(x) = x on [2, 4] represented as Chebyshev on that domain
        // T_1(t) = t, mapped: f(x) = (b-a)/2 * t + (a+b)/2 = t + 3
        // where t = (2x-6)/2 = x-3
        // So f(x) = (x-3) + 3 = x ✓ if we use coeffs for t
        //
        // Actually, to represent f(x)=x on [2,4], we need:
        // f(x) = c_0/2 + c_1*T_1(t) where t = (2x-6)/2 = x-3
        // f(x) = c_0/2 + c_1*(x-3)
        // For f(x) = x: c_0/2 = 3, c_1 = 1 → c_0 = 6
        let seg = ChebyshevSegment {
            coeffs: vec![6.0, 1.0],
            a: 2.0,
            b: 4.0,
        };
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(3.0).unwrap();
        let f = seg.build_graph(&mut ad, x);

        // f(3) should be 3
        assert!(
            (ad.eval(f).unwrap() - 3.0).abs() < 1e-12,
            "f(3) = {}",
            ad.eval(f).unwrap()
        );

        // f'(x) = 1
        let dfdx = ad.differentiate(f, x).unwrap();
        assert!(
            (ad.eval(dfdx).unwrap() - 1.0).abs() < 1e-12,
            "f'(3) = {}",
            ad.eval(dfdx).unwrap()
        );
    }

    #[test]
    fn graph_sin_fit_derivative() {
        // Fit sin(x) on [0, π], then differentiate via the graph
        let n = 100;
        let x_data: Vec<f64> = (0..=n)
            .map(|i| std::f64::consts::PI * i as f64 / n as f64)
            .collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
        let result = fit_dense(
            &x_data,
            &y_data,
            &[0.0, std::f64::consts::PI],
            &FitOptions { degree: 20 },
        )
        .unwrap();

        let seg = result.fit.segment(0);
        let test_x = 1.0;

        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(test_x).unwrap();
        let f = seg.build_graph(&mut ad, x);

        // Check value (tolerance accounts for linear interpolation resampling)
        let val = ad.eval(f).unwrap();
        assert!(
            (val - test_x.sin()).abs() < 1e-4,
            "f({test_x}) = {val}, expected {}",
            test_x.sin()
        );

        // Check derivative
        let dfdx = ad.differentiate(f, x).unwrap();
        let deriv = ad.eval(dfdx).unwrap();
        assert!(
            (deriv - test_x.cos()).abs() < 1e-3,
            "f'({test_x}) = {deriv}, expected {}",
            test_x.cos()
        );
    }

    #[test]
    fn graph_second_derivative() {
        // f(x) = sin(x) on [0, π], f''(x) = -sin(x)
        let n = 100;
        let x_data: Vec<f64> = (0..=n)
            .map(|i| std::f64::consts::PI * i as f64 / n as f64)
            .collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
        let result = fit_dense(
            &x_data,
            &y_data,
            &[0.0, std::f64::consts::PI],
            &FitOptions { degree: 20 },
        )
        .unwrap();

        let seg = result.fit.segment(0);
        let test_x = 1.0;

        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(test_x).unwrap();
        let f = seg.build_graph(&mut ad, x);
        let dfdx = ad.differentiate(f, x).unwrap();
        let d2fdx2 = ad.differentiate(dfdx, x).unwrap();

        // Dense fit of sin(x) degree 20: f'' error dominated by
        // resampling noise amplified twice. Observed: ~2.8e-3 at x=1.0
        let second_deriv = ad.eval(d2fdx2).unwrap();
        let expected = -test_x.sin();
        assert!(
            (second_deriv - expected).abs() < 1e-2,
            "f''({test_x}) = {second_deriv}, expected {expected}, err = {:.2e}",
            (second_deriv - expected).abs()
        );
    }

    #[test]
    fn graph_compose_with_ad_ops() {
        // Build Clenshaw graph for a fit, then compose: g = exp(f(x))
        // g'(x) = exp(f(x)) * f'(x)
        let n = 50;
        let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x).collect(); // f=x²
        let result = fit_dense(&x_data, &y_data, &[0.0, 1.0], &FitOptions { degree: 10 }).unwrap();

        let seg = result.fit.segment(0);
        let test_x = 0.5;

        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(test_x).unwrap();
        let f = seg.build_graph(&mut ad, x);
        let g = ad.exp(f); // g = exp(x²)

        // Observed: val err ~3e-4, deriv err ~2e-3
        let val = ad.eval(g).unwrap();
        let expected = (test_x * test_x).exp();
        assert!(
            (val - expected).abs() < 1e-3,
            "g({test_x}) = {val}, expected {expected}, err = {:.2e}",
            (val - expected).abs()
        );

        let dgdx = ad.differentiate(g, x).unwrap();
        let deriv = ad.eval(dgdx).unwrap();
        // g'(x) = exp(x²) * 2x
        let expected_deriv = expected * 2.0 * test_x;
        assert!(
            (deriv - expected_deriv).abs() < 5e-3,
            "g'({test_x}) = {deriv}, expected {expected_deriv}, err = {:.2e}",
            (deriv - expected_deriv).abs()
        );
    }

    #[test]
    fn piecewise_build_segment_graph_out_of_range() {
        let seg = ChebyshevSegment {
            coeffs: vec![2.0],
            a: 0.0,
            b: 1.0,
        };
        let pw = PiecewiseFit::new(vec![seg], vec![0.0, 1.0]);
        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(0.5).unwrap();
        assert!(pw.build_segment_graph(&mut ad, x, 1).is_err());
    }
}
