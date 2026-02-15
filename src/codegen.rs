//! WGSL code generation from compiled graphs.
//!
//! Generates standalone WGSL functions from a [`CompiledGraph`]. The output
//! is a struct definition and a pure function that can be embedded in any
//! WGSL shader — no runtime dependency on the `wgpu` feature.
//!
//! # Example
//!
//! ```
//! use bevy_autodiff::AutoDiff;
//!
//! let mut ad = AutoDiff::new();
//! let x = ad.var(0.0).unwrap();
//! let y = ad.var(0.0).unwrap();
//! let f = ad.mul(x, y);
//! let graph = ad.compile_order(f, &[x, y], 1).unwrap();
//!
//! let wgsl = graph.to_wgsl("mul_xy").unwrap();
//! assert!(wgsl.contains("struct MulXyOutput"));
//! assert!(wgsl.contains("fn mul_xy("));
//! ```

use std::fmt::Write;

use crate::compiled::{CompiledGraph, NodeOp};
use crate::components::{BinaryOp, UnaryOp};

impl CompiledGraph {
    /// Generates a standalone WGSL function from this compiled graph.
    ///
    /// Returns a string containing:
    /// 1. A result struct `{FuncName}Output` with fields for the primal value
    ///    and each compiled partial derivative
    /// 2. A function `{func_name}(p0: f32, p1: f32, ...) -> {FuncName}Output`
    ///
    /// The generated code uses direct WGSL expressions (no interpreter loop),
    /// making it suitable for embedding in custom compute or fragment shaders.
    ///
    /// # Parameters
    ///
    /// - `func_name`: The name for the generated function (snake_case recommended).
    ///   The struct name is derived by converting to PascalCase and appending "Output".
    ///
    /// # Errors
    ///
    /// Returns [`NonFiniteWgsl`](crate::error::AutoDiffError::NonFiniteWgsl) if the graph
    /// contains a constant with a non-finite value (NaN or infinity).
    pub fn to_wgsl(&self, func_name: &str) -> Result<String, crate::error::AutoDiffError> {
        let nodes = self.nodes();
        let output_index = self.output_index();
        let partial_outputs = self.partial_outputs();

        let struct_name = to_pascal_case(func_name);
        let struct_name = format!("{struct_name}Output");

        let mut out = String::new();

        // --- Struct definition ---
        // writeln! on String is infallible
        let _ = writeln!(out, "struct {struct_name} {{");
        let _ = writeln!(out, "    value: f32,");
        for (multi_index, _) in partial_outputs {
            let field = partial_field_name(multi_index);
            let _ = writeln!(out, "    {field}: f32,");
        }
        let _ = writeln!(out, "}}");
        let _ = writeln!(out);

        // --- Function signature ---
        let params: Vec<String> = (0..self.num_inputs())
            .map(|i| format!("p{i}: f32"))
            .collect();
        let _ = writeln!(out, "fn {func_name}({}) -> {struct_name} {{", params.join(", "));

        // --- Node statements ---
        for (i, node) in nodes.iter().enumerate() {
            let stmt = node_to_wgsl(i, node)?;
            let _ = writeln!(out, "    {stmt}");
        }

        // --- Return statement ---
        let mut return_fields = vec![format!("v{output_index}")];
        for (_, node_idx) in partial_outputs {
            return_fields.push(format!("v{node_idx}"));
        }
        let _ = writeln!(out, "    return {struct_name}({});", return_fields.join(", "));
        let _ = writeln!(out, "}}");

        Ok(out)
    }
}

/// Convert a single `NodeOp` to a WGSL `let` statement.
fn node_to_wgsl(index: usize, node: &NodeOp) -> Result<String, crate::error::AutoDiffError> {
    match *node {
        NodeOp::Input(pos) => Ok(format!("let v{index} = p{pos};")),
        NodeOp::Constant(val) => Ok(format!("let v{index} = {};", format_f32(val)?)),
        NodeOp::Unary { op, src } => {
            let expr = unary_to_wgsl(op, &format!("v{src}"));
            Ok(format!("let v{index} = {expr};"))
        }
        NodeOp::Binary { op, lhs, rhs } => {
            let expr = binary_to_wgsl(op, &format!("v{lhs}"), &format!("v{rhs}"));
            Ok(format!("let v{index} = {expr};"))
        }
    }
}

/// Map a `UnaryOp` to the WGSL expression.
fn unary_to_wgsl(op: UnaryOp, src: &str) -> String {
    match op {
        UnaryOp::Neg => format!("-({src})"),
        UnaryOp::Sin => format!("sin({src})"),
        UnaryOp::Cos => format!("cos({src})"),
        UnaryOp::Tan => format!("tan({src})"),
        UnaryOp::Exp => format!("exp({src})"),
        UnaryOp::Ln => format!("log({src})"),
        UnaryOp::Sqrt => format!("sqrt({src})"),
        UnaryOp::Sinh => format!("sinh({src})"),
        UnaryOp::Cosh => format!("cosh({src})"),
        UnaryOp::Tanh => format!("tanh({src})"),
        UnaryOp::Asin => format!("asin({src})"),
        UnaryOp::Acos => format!("acos({src})"),
        UnaryOp::Atan => format!("atan({src})"),
        UnaryOp::Asinh => format!("asinh({src})"),
        UnaryOp::Acosh => format!("acosh({src})"),
        UnaryOp::Atanh => format!("atanh({src})"),
    }
}

/// Map a `BinaryOp` to the WGSL expression.
fn binary_to_wgsl(op: BinaryOp, lhs: &str, rhs: &str) -> String {
    match op {
        BinaryOp::Add => format!("{lhs} + {rhs}"),
        BinaryOp::Sub => format!("{lhs} - {rhs}"),
        BinaryOp::Mul => format!("{lhs} * {rhs}"),
        BinaryOp::Div | BinaryOp::DivLog => format!("{lhs} / {rhs}"),
        BinaryOp::Pow | BinaryOp::PowLog => format!("pow({lhs}, {rhs})"),
    }
}

/// Format an f64 constant as an f32 WGSL literal.
///
/// # Errors
///
/// Returns [`NonFiniteWgsl`](crate::error::AutoDiffError::NonFiniteWgsl) if the value is
/// NaN or infinite, which are not valid WGSL literals.
fn format_f32(val: f64) -> Result<String, crate::error::AutoDiffError> {
    if !val.is_finite() {
        return Err(crate::error::AutoDiffError::NonFiniteWgsl { value: val });
    }
    let f = val as f32;
    // Ensure the literal always has a decimal point so WGSL treats it as f32
    let s = format!("{f}");
    if s.contains('.') {
        Ok(s)
    } else {
        Ok(format!("{s}.0"))
    }
}

/// Convert a snake_case string to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(c) => {
                    let mut s = c.to_uppercase().to_string();
                    s.extend(chars);
                    s
                }
                None => String::new(),
            }
        })
        .collect()
}

/// Generate a struct field name from a multi-index, e.g. `[1, 0]` → `d1_0`.
fn partial_field_name(multi_index: &[usize]) -> String {
    let parts: Vec<String> = multi_index.iter().map(|i| i.to_string()).collect();
    format!("d{}", parts.join("_"))
}

#[cfg(test)]
mod tests {
    use crate::compiled::CompiledGraph;
    use crate::components::{BinaryOp, UnaryOp};
    use crate::compiled::NodeOp;

    use super::*;

    #[test]
    fn constant_only() {
        let nodes = vec![NodeOp::Constant(3.14)];
        let cg = CompiledGraph::new(nodes, 0, 0, vec![]);
        let wgsl = cg.to_wgsl("pi").unwrap();
        assert!(wgsl.contains("struct PiOutput"));
        assert!(wgsl.contains("fn pi() -> PiOutput"));
        assert!(wgsl.contains("let v0 = 3.14;"));
        assert!(wgsl.contains("return PiOutput(v0);"));
    }

    #[test]
    fn identity() {
        // f(x) = x
        let nodes = vec![NodeOp::Input(0)];
        let cg = CompiledGraph::new(nodes, 1, 0, vec![]);
        let wgsl = cg.to_wgsl("identity").unwrap();
        assert!(wgsl.contains("fn identity(p0: f32) -> IdentityOutput"));
        assert!(wgsl.contains("let v0 = p0;"));
        assert!(wgsl.contains("return IdentityOutput(v0);"));
    }

    #[test]
    fn linear() {
        // f(x) = 2*x + 1
        let nodes = vec![
            NodeOp::Input(0),
            NodeOp::Constant(2.0),
            NodeOp::Binary { op: BinaryOp::Mul, lhs: 1, rhs: 0 },
            NodeOp::Constant(1.0),
            NodeOp::Binary { op: BinaryOp::Add, lhs: 2, rhs: 3 },
        ];
        let cg = CompiledGraph::new(nodes, 1, 4, vec![]);
        let wgsl = cg.to_wgsl("linear").unwrap();
        assert!(wgsl.contains("let v2 = v1 * v0;"));
        assert!(wgsl.contains("let v4 = v2 + v3;"));
        assert!(wgsl.contains("return LinearOutput(v4);"));
    }

    #[test]
    fn unary_ops() {
        let ops_and_wgsl = [
            (UnaryOp::Neg, "-(v0)"),
            (UnaryOp::Sin, "sin(v0)"),
            (UnaryOp::Cos, "cos(v0)"),
            (UnaryOp::Tan, "tan(v0)"),
            (UnaryOp::Exp, "exp(v0)"),
            (UnaryOp::Ln, "log(v0)"),
            (UnaryOp::Sqrt, "sqrt(v0)"),
            (UnaryOp::Sinh, "sinh(v0)"),
            (UnaryOp::Cosh, "cosh(v0)"),
            (UnaryOp::Tanh, "tanh(v0)"),
            (UnaryOp::Asin, "asin(v0)"),
            (UnaryOp::Acos, "acos(v0)"),
            (UnaryOp::Atan, "atan(v0)"),
            (UnaryOp::Asinh, "asinh(v0)"),
            (UnaryOp::Acosh, "acosh(v0)"),
            (UnaryOp::Atanh, "atanh(v0)"),
        ];

        for (op, expected_expr) in ops_and_wgsl {
            let nodes = vec![
                NodeOp::Input(0),
                NodeOp::Unary { op, src: 0 },
            ];
            let cg = CompiledGraph::new(nodes, 1, 1, vec![]);
            let wgsl = cg.to_wgsl("f").unwrap();
            assert!(
                wgsl.contains(&format!("let v1 = {expected_expr};")),
                "op {:?}: expected '{}' in:\n{}",
                op, expected_expr, wgsl
            );
        }
    }

    #[test]
    fn binary_ops() {
        let ops_and_wgsl = [
            (BinaryOp::Add, "v0 + v1"),
            (BinaryOp::Sub, "v0 - v1"),
            (BinaryOp::Mul, "v0 * v1"),
            (BinaryOp::Div, "v0 / v1"),
            (BinaryOp::Pow, "pow(v0, v1)"),
        ];

        for (op, expected_expr) in ops_and_wgsl {
            let nodes = vec![
                NodeOp::Input(0),
                NodeOp::Input(1),
                NodeOp::Binary { op, lhs: 0, rhs: 1 },
            ];
            let cg = CompiledGraph::new(nodes, 2, 2, vec![]);
            let wgsl = cg.to_wgsl("f").unwrap();
            assert!(
                wgsl.contains(&format!("let v2 = {expected_expr};")),
                "op {:?}: expected '{}' in:\n{}",
                op, expected_expr, wgsl
            );
        }
    }

    #[test]
    fn with_partials() {
        // f(x) = x^2, df/dx = 2x (manually constructed)
        let nodes = vec![
            NodeOp::Input(0),       // v0: x
            NodeOp::Binary {        // v1: x * x
                op: BinaryOp::Mul,
                lhs: 0,
                rhs: 0,
            },
            NodeOp::Constant(2.0),  // v2: 2
            NodeOp::Binary {        // v3: 2 * x
                op: BinaryOp::Mul,
                lhs: 2,
                rhs: 0,
            },
        ];
        let cg = CompiledGraph::new(nodes, 1, 1, vec![(vec![1], 3)]);
        let wgsl = cg.to_wgsl("square").unwrap();

        // Struct should have value + partial field
        assert!(wgsl.contains("value: f32,"));
        assert!(wgsl.contains("d1: f32,"));
        // Return should include both
        assert!(wgsl.contains("return SquareOutput(v1, v3);"));
    }

    #[test]
    fn two_inputs() {
        // f(x, y) = x * y
        let nodes = vec![
            NodeOp::Input(0),
            NodeOp::Input(1),
            NodeOp::Binary { op: BinaryOp::Mul, lhs: 0, rhs: 1 },
        ];
        let cg = CompiledGraph::new(nodes, 2, 2, vec![]);
        let wgsl = cg.to_wgsl("mul_xy").unwrap();

        assert!(wgsl.contains("fn mul_xy(p0: f32, p1: f32) -> MulXyOutput"));
        assert!(wgsl.contains("let v0 = p0;"));
        assert!(wgsl.contains("let v1 = p1;"));
    }

    #[test]
    fn validate_output_parseable() {
        // Verify the generated WGSL has the expected structure tokens
        let nodes = vec![
            NodeOp::Input(0),
            NodeOp::Unary { op: UnaryOp::Sin, src: 0 },
        ];
        let cg = CompiledGraph::new(nodes, 1, 1, vec![]);
        let wgsl = cg.to_wgsl("my_func").unwrap();

        // Must contain struct definition
        assert!(wgsl.contains("struct MyFuncOutput {"));
        // Must contain function definition
        assert!(wgsl.contains("fn my_func("));
        // Must contain return with struct constructor
        assert!(wgsl.contains("return MyFuncOutput("));
        // Must contain closing braces
        assert_eq!(wgsl.matches('}').count(), 2); // struct + fn
    }

    #[test]
    fn multi_index_field_names() {
        // f(x, y) with partials [1,0] and [0,1]
        let nodes = vec![
            NodeOp::Input(0),
            NodeOp::Input(1),
            NodeOp::Binary { op: BinaryOp::Mul, lhs: 0, rhs: 1 },
            NodeOp::Input(1), // stand-in for df/dx node
            NodeOp::Input(0), // stand-in for df/dy node
        ];
        let cg = CompiledGraph::new(
            nodes,
            2,
            2,
            vec![(vec![1, 0], 3), (vec![0, 1], 4)],
        );
        let wgsl = cg.to_wgsl("f").unwrap();
        assert!(wgsl.contains("d1_0: f32,"));
        assert!(wgsl.contains("d0_1: f32,"));
    }

    #[test]
    fn end_to_end_via_autodiff() {
        // Test using the actual AutoDiff API
        use crate::AutoDiff;

        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let f = ad.sin(x);
        let graph = ad.compile_order(f, &[x], 1).unwrap();

        let wgsl = graph.to_wgsl("sin_x").unwrap();
        assert!(wgsl.contains("struct SinXOutput {"));
        assert!(wgsl.contains("fn sin_x(p0: f32) -> SinXOutput {"));
        assert!(wgsl.contains("value: f32,"));
        assert!(wgsl.contains("d1: f32,"));
        // Should contain sin and cos (derivative of sin)
        assert!(wgsl.contains("sin("));
        assert!(wgsl.contains("cos("));
    }

    #[test]
    fn pascal_case_conversion() {
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("sin_xy"), "SinXy");
        assert_eq!(to_pascal_case("f"), "F");
        assert_eq!(to_pascal_case("my_long_name"), "MyLongName");
    }

    #[test]
    fn constant_formatting() {
        // Ensure integer-valued floats get a decimal point
        assert!(format_f32(1.0).unwrap().contains('.'));
        assert!(format_f32(0.0).unwrap().contains('.'));
        // Non-integer already has decimal
        assert!(format_f32(3.14).unwrap().contains('.'));
    }

    #[test]
    fn constant_nan_returns_error() {
        assert!(format_f32(f64::NAN).is_err());
    }

    #[test]
    fn constant_inf_returns_error() {
        assert!(format_f32(f64::INFINITY).is_err());
    }

    // =========================================================================
    // f32 precision diagnostic tests
    // =========================================================================

    /// Evaluate a NodeOp array using f32 arithmetic (simulating WGSL precision).
    fn eval_f32(nodes: &[NodeOp], inputs: &[f32]) -> Vec<f32> {
        let mut values = vec![0.0f32; nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            values[i] = match *node {
                NodeOp::Input(idx) => inputs[idx],
                NodeOp::Constant(v) => v as f32,
                NodeOp::Unary { op, src } => {
                    let x = values[src];
                    match op {
                        UnaryOp::Neg => -x,
                        UnaryOp::Sin => x.sin(),
                        UnaryOp::Cos => x.cos(),
                        UnaryOp::Tan => x.tan(),
                        UnaryOp::Exp => x.exp(),
                        UnaryOp::Ln => x.ln(),
                        UnaryOp::Sqrt => x.sqrt(),
                        UnaryOp::Sinh => x.sinh(),
                        UnaryOp::Cosh => x.cosh(),
                        UnaryOp::Tanh => x.tanh(),
                        UnaryOp::Asin => x.asin(),
                        UnaryOp::Acos => x.acos(),
                        UnaryOp::Atan => x.atan(),
                        UnaryOp::Asinh => x.asinh(),
                        UnaryOp::Acosh => x.acosh(),
                        UnaryOp::Atanh => x.atanh(),
                    }
                }
                NodeOp::Binary { op, lhs, rhs } => {
                    let a = values[lhs];
                    let b = values[rhs];
                    match op {
                        BinaryOp::Add => a + b,
                        BinaryOp::Sub => a - b,
                        BinaryOp::Mul => a * b,
                        BinaryOp::Div | BinaryOp::DivLog => a / b,
                        BinaryOp::Pow | BinaryOp::PowLog => a.powf(b),
                    }
                }
            };
        }
        values
    }

    /// Simple second-order test: f(x) = x³, f'(x) = 3x², f''(x) = 6x.
    /// This verifies the codegen structure is correct for order-2 partials.
    #[test]
    fn second_order_simple_cubic() {
        use crate::AutoDiff;

        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let x2 = ad.mul(x, x);
        let f = ad.mul(x2, x); // x³

        let mut graph = ad.compile_order(f, &[x], 2).unwrap();

        // f64 eval
        graph.eval(&[2.0]).unwrap();
        let val_f64 = graph.value();
        let d1_f64 = graph.partial(&[1]).unwrap();
        let d2_f64 = graph.partial(&[2]).unwrap();

        assert!((val_f64 - 8.0).abs() < 1e-10, "f(2) = 8, got {val_f64}");
        assert!((d1_f64 - 12.0).abs() < 1e-10, "f'(2) = 12, got {d1_f64}");
        assert!((d2_f64 - 12.0).abs() < 1e-10, "f''(2) = 12, got {d2_f64}");

        // f32 simulation
        let nodes = graph.nodes();
        let output_index = graph.output_index();
        let partial_outputs = graph.partial_outputs();

        let vals_f32 = eval_f32(nodes, &[2.0f32]);
        let val_f32 = vals_f32[output_index];
        let d1_idx = partial_outputs.iter().find(|(mi, _)| mi == &[1]).unwrap().1;
        let d2_idx = partial_outputs.iter().find(|(mi, _)| mi == &[2]).unwrap().1;
        let d1_f32 = vals_f32[d1_idx];
        let d2_f32 = vals_f32[d2_idx];

        assert!((val_f32 - 8.0).abs() < 1e-4, "f32: f(2) = 8, got {val_f32}");
        assert!((d1_f32 - 12.0).abs() < 1e-4, "f32: f'(2) = 12, got {d1_f32}");
        assert!((d2_f32 - 12.0).abs() < 1e-4, "f32: f''(2) = 12, got {d2_f32}");
    }

    /// Two-body gravitational Hessian diagnostic.
    ///
    /// Builds a_x = -mu * x / r³ and compiles at order 2.
    /// Compares f64 eval, f32 simulation, and analytical Hessian to determine
    /// whether second-order WGSL errors are due to codegen bugs or f32 precision.
    #[test]
    fn second_order_two_body_f32_diagnostic() {
        use crate::AutoDiff;

        let mut ad = AutoDiff::new();
        let rx = ad.var(0.0).unwrap();
        let ry = ad.var(0.0).unwrap();
        let rz = ad.var(0.0).unwrap();

        let rx2 = ad.mul(rx, rx);
        let ry2 = ad.mul(ry, ry);
        let rz2 = ad.mul(rz, rz);
        let rxy2 = ad.add(rx2, ry2);
        let r2 = ad.add(rxy2, rz2);
        let r_mag = ad.sqrt(r2);
        let r3 = ad.mul(r2, r_mag);
        let neg_mu = ad.constant(-398600.4418);
        let factor = ad.div(neg_mu, r3);
        let accel = ad.mul(factor, rx);

        let mut graph = ad.compile_order(accel, &[rx, ry, rz], 2).unwrap();

        let pos = [5000.0, 3000.0, 1000.0];

        // --- f64 eval (ground truth from CompiledGraph) ---
        graph.eval(&pos).unwrap();
        let val_f64 = graph.value();
        let d1_0_0_f64 = graph.partial(&[1, 0, 0]).unwrap();
        let d0_1_0_f64 = graph.partial(&[0, 1, 0]).unwrap();
        let d0_0_1_f64 = graph.partial(&[0, 0, 1]).unwrap();
        let d2_0_0_f64 = graph.partial(&[2, 0, 0]).unwrap();
        let d1_1_0_f64 = graph.partial(&[1, 1, 0]).unwrap();
        let d1_0_1_f64 = graph.partial(&[1, 0, 1]).unwrap();
        let d0_2_0_f64 = graph.partial(&[0, 2, 0]).unwrap();
        let d0_1_1_f64 = graph.partial(&[0, 1, 1]).unwrap();
        let d0_0_2_f64 = graph.partial(&[0, 0, 2]).unwrap();

        // --- Analytical Hessian for validation ---
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let mu = 398600.4418;
        let r2_val = x * x + y * y + z * z;
        let r = r2_val.sqrt();
        let r7 = r.powi(7);
        // a_x = -mu * x / r³
        // da_x/dr_j = mu * (3*x*r_j/r⁵ - delta(x,j)/r³)
        // d²a_x/dr_j dr_k = mu/r⁷ * [-15*x*r_j*r_k + 3*r²*(r_k*d(x,j) + r_j*d(x,k) + x*d(j,k))]
        let d = |i: usize, j: usize| -> f64 { if i == j { 1.0 } else { 0.0 } };
        let ri = [x, y, z];
        let analytical_hessian = |j: usize, k: usize| -> f64 {
            (mu / r7)
                * (-15.0 * x * ri[j] * ri[k]
                    + 3.0 * r2_val * (ri[k] * d(0, j) + ri[j] * d(0, k) + x * d(j, k)))
        };

        // Verify f64 eval matches analytical
        let h00_analytical = analytical_hessian(0, 0);
        let h01_analytical = analytical_hessian(0, 1);
        let h02_analytical = analytical_hessian(0, 2);
        let h11_analytical = analytical_hessian(1, 1);
        let h12_analytical = analytical_hessian(1, 2);
        let h22_analytical = analytical_hessian(2, 2);

        let rel_err = |a: f64, b: f64| -> f64 {
            if b.abs() < 1e-30 { a.abs() } else { (a - b).abs() / b.abs() }
        };

        eprintln!("\n=== Two-body Hessian diagnostic ===");
        eprintln!("Position: [{x}, {y}, {z}]");
        eprintln!("r = {r:.6}, num_nodes = {}", graph.num_nodes());
        eprintln!();

        // f64 vs analytical
        eprintln!("--- f64 eval vs analytical ---");
        eprintln!("d2_0_0: f64={d2_0_0_f64:>15.6e}  analytical={h00_analytical:>15.6e}  rel_err={:.2e}", rel_err(d2_0_0_f64, h00_analytical));
        eprintln!("d1_1_0: f64={d1_1_0_f64:>15.6e}  analytical={h01_analytical:>15.6e}  rel_err={:.2e}", rel_err(d1_1_0_f64, h01_analytical));
        eprintln!("d1_0_1: f64={d1_0_1_f64:>15.6e}  analytical={h02_analytical:>15.6e}  rel_err={:.2e}", rel_err(d1_0_1_f64, h02_analytical));
        eprintln!("d0_2_0: f64={d0_2_0_f64:>15.6e}  analytical={h11_analytical:>15.6e}  rel_err={:.2e}", rel_err(d0_2_0_f64, h11_analytical));
        eprintln!("d0_1_1: f64={d0_1_1_f64:>15.6e}  analytical={h12_analytical:>15.6e}  rel_err={:.2e}", rel_err(d0_1_1_f64, h12_analytical));
        eprintln!("d0_0_2: f64={d0_0_2_f64:>15.6e}  analytical={h22_analytical:>15.6e}  rel_err={:.2e}", rel_err(d0_0_2_f64, h22_analytical));
        eprintln!();

        // --- f32 simulation ---
        let nodes = graph.nodes();
        let output_index = graph.output_index();
        let partial_outputs = graph.partial_outputs();

        let pos_f32: Vec<f32> = pos.iter().map(|&v| v as f32).collect();
        let vals_f32 = eval_f32(nodes, &pos_f32);

        let find_partial = |mi: &[usize]| -> f32 {
            let idx = partial_outputs.iter().find(|(m, _)| m == mi).unwrap().1;
            vals_f32[idx]
        };

        let val_f32 = vals_f32[output_index];
        let d1_0_0_f32 = find_partial(&[1, 0, 0]);
        let d0_1_0_f32 = find_partial(&[0, 1, 0]);
        let d0_0_1_f32 = find_partial(&[0, 0, 1]);
        let d2_0_0_f32 = find_partial(&[2, 0, 0]);
        let d1_1_0_f32 = find_partial(&[1, 1, 0]);
        let d1_0_1_f32 = find_partial(&[1, 0, 1]);
        let d0_2_0_f32 = find_partial(&[0, 2, 0]);
        let d0_1_1_f32 = find_partial(&[0, 1, 1]);
        let d0_0_2_f32 = find_partial(&[0, 0, 2]);

        eprintln!("--- f32 simulation vs analytical ---");
        eprintln!("value:  f64={val_f64:>15.6e}  f32={val_f32:>15.6e}");
        eprintln!();
        eprintln!("First derivatives:");
        eprintln!("d1_0_0: f64={d1_0_0_f64:>15.6e}  f32={d1_0_0_f32:>15.6e}  f32_rel_err={:.2e}", rel_err(d1_0_0_f32 as f64, d1_0_0_f64));
        eprintln!("d0_1_0: f64={d0_1_0_f64:>15.6e}  f32={d0_1_0_f32:>15.6e}  f32_rel_err={:.2e}", rel_err(d0_1_0_f32 as f64, d0_1_0_f64));
        eprintln!("d0_0_1: f64={d0_0_1_f64:>15.6e}  f32={d0_0_1_f32:>15.6e}  f32_rel_err={:.2e}", rel_err(d0_0_1_f32 as f64, d0_0_1_f64));
        eprintln!();
        eprintln!("Second derivatives (THE KEY COMPARISON):");
        eprintln!("d2_0_0: f64={d2_0_0_f64:>15.6e}  f32={d2_0_0_f32:>15.6e}  analytical={h00_analytical:>15.6e}  f32_rel_err={:.2e}", rel_err(d2_0_0_f32 as f64, h00_analytical));
        eprintln!("d1_1_0: f64={d1_1_0_f64:>15.6e}  f32={d1_1_0_f32:>15.6e}  analytical={h01_analytical:>15.6e}  f32_rel_err={:.2e}", rel_err(d1_1_0_f32 as f64, h01_analytical));
        eprintln!("d1_0_1: f64={d1_0_1_f64:>15.6e}  f32={d1_0_1_f32:>15.6e}  analytical={h02_analytical:>15.6e}  f32_rel_err={:.2e}", rel_err(d1_0_1_f32 as f64, h02_analytical));
        eprintln!("d0_2_0: f64={d0_2_0_f64:>15.6e}  f32={d0_2_0_f32:>15.6e}  analytical={h11_analytical:>15.6e}  f32_rel_err={:.2e}", rel_err(d0_2_0_f32 as f64, h11_analytical));
        eprintln!("d0_1_1: f64={d0_1_1_f64:>15.6e}  f32={d0_1_1_f32:>15.6e}  analytical={h12_analytical:>15.6e}  f32_rel_err={:.2e}", rel_err(d0_1_1_f32 as f64, h12_analytical));
        eprintln!("d0_0_2: f64={d0_0_2_f64:>15.6e}  f32={d0_0_2_f32:>15.6e}  analytical={h22_analytical:>15.6e}  f32_rel_err={:.2e}", rel_err(d0_0_2_f32 as f64, h22_analytical));
        eprintln!();

        // Check if f32 simulation matches the bug report pattern
        let f32_matches_bug_report =
            rel_err(d2_0_0_f32 as f64, h00_analytical) > 0.5
            || rel_err(d0_2_0_f32 as f64, h11_analytical) > 0.5
            || rel_err(d0_0_2_f32 as f64, h22_analytical) > 0.5;

        if f32_matches_bug_report {
            eprintln!("DIAGNOSIS: f32 Rust simulation reproduces the bug report errors.");
            eprintln!("This is catastrophic cancellation in f32, NOT a codegen bug.");
            eprintln!("The to_wgsl() code generation is structurally correct.");
        } else {
            eprintln!("DIAGNOSIS: f32 Rust simulation does NOT reproduce the errors.");
            eprintln!("This suggests a genuine codegen bug in to_wgsl().");
        }
    }

    /// First-order: pow_log derivative matches pow derivative numerically.
    #[test]
    fn pow_log_first_order_matches_pow() {
        use crate::AutoDiff;

        for &x_val in &[0.5, 1.0, 2.0, 4.0] {
            for &p in &[-3.0, -1.5, 0.5, 2.0, 2.5] {
                let mut ad_pow = AutoDiff::new();
                let x1 = ad_pow.var(x_val).unwrap();
                let f1 = ad_pow.powf(x1, p);
                let df1 = ad_pow.differentiate(f1, x1).unwrap();
                let deriv_pow = ad_pow.eval(df1).unwrap();

                let mut ad_log = AutoDiff::new();
                let x2 = ad_log.var(x_val).unwrap();
                let f2 = ad_log.powf_log(x2, p);
                let df2 = ad_log.differentiate(f2, x2).unwrap();
                let deriv_log = ad_log.eval(df2).unwrap();

                assert!(
                    (deriv_pow - deriv_log).abs() < 1e-10 * deriv_pow.abs().max(1.0),
                    "pow_log first-order mismatch at x={x_val}, p={p}: pow={deriv_pow}, log={deriv_log}"
                );
            }
        }
    }

    /// div_log first-order: matches div derivative numerically.
    #[test]
    fn div_log_first_order_matches_div() {
        use crate::AutoDiff;

        for &x_val in &[1.0, 2.0, 3.0] {
            for &y_val in &[0.5, 1.0, 2.0] {
                // d(x/y)/dx = 1/y
                let mut ad_div = AutoDiff::new();
                let x1 = ad_div.var(x_val).unwrap();
                let y1 = ad_div.var(y_val).unwrap();
                let f1 = ad_div.div(x1, y1);
                let df1_dx = ad_div.differentiate(f1, x1).unwrap();
                let df1_dy = ad_div.differentiate(f1, y1).unwrap();
                let deriv_div_dx = ad_div.eval(df1_dx).unwrap();
                let deriv_div_dy = ad_div.eval(df1_dy).unwrap();

                let mut ad_log = AutoDiff::new();
                let x2 = ad_log.var(x_val).unwrap();
                let y2 = ad_log.var(y_val).unwrap();
                let f2 = ad_log.div_log(x2, y2);
                let df2_dx = ad_log.differentiate(f2, x2).unwrap();
                let df2_dy = ad_log.differentiate(f2, y2).unwrap();
                let deriv_log_dx = ad_log.eval(df2_dx).unwrap();
                let deriv_log_dy = ad_log.eval(df2_dy).unwrap();

                assert!(
                    (deriv_div_dx - deriv_log_dx).abs() < 1e-10 * deriv_div_dx.abs().max(1.0),
                    "div_log dx mismatch at x={x_val},y={y_val}: div={deriv_div_dx}, log={deriv_log_dx}"
                );
                assert!(
                    (deriv_div_dy - deriv_log_dy).abs() < 1e-10 * deriv_div_dy.abs().max(1.0),
                    "div_log dy mismatch at x={x_val},y={y_val}: div={deriv_div_dy}, log={deriv_log_dy}"
                );
            }
        }
    }

    /// powi_log derivative: d/dx(x^3) = 3x² via powi_log.
    #[test]
    fn powi_log_derivative() {
        use crate::AutoDiff;

        for &x_val in &[1.0, 2.0, 3.0] {
            let mut ad = AutoDiff::new();
            let x = ad.var(x_val).unwrap();
            let f = ad.powi_log(x, 3);
            let df = ad.differentiate(f, x).unwrap();
            let deriv = ad.eval(df).unwrap();
            let expected = 3.0 * x_val * x_val;

            assert!(
                (deriv - expected).abs() < 1e-10,
                "powi_log d/dx(x^3) at x={x_val}: got {deriv}, expected {expected}"
            );
        }
    }

    /// Second-order pow_log derivative matches pow derivative numerically.
    #[test]
    fn pow_log_second_order_matches_pow() {
        use crate::AutoDiff;

        for &x_val in &[0.5, 1.0, 2.0, 4.0] {
            for &p in &[-3.0, -1.5, 2.5] {
                let mut ad_pow = AutoDiff::new();
                let x1 = ad_pow.var(x_val).unwrap();
                let f1 = ad_pow.powf(x1, p);
                let df1 = ad_pow.differentiate(f1, x1).unwrap();
                let d2f1 = ad_pow.differentiate(df1, x1).unwrap();
                let d2_pow = ad_pow.eval(d2f1).unwrap();

                let mut ad_log = AutoDiff::new();
                let x2 = ad_log.var(x_val).unwrap();
                let f2 = ad_log.powf_log(x2, p);
                let df2 = ad_log.differentiate(f2, x2).unwrap();
                let d2f2 = ad_log.differentiate(df2, x2).unwrap();
                let d2_log = ad_log.eval(d2f2).unwrap();

                assert!(
                    (d2_pow - d2_log).abs() < 1e-8 * d2_pow.abs().max(1.0),
                    "pow_log 2nd-order mismatch at x={x_val}, p={p}: pow={d2_pow}, log={d2_log}"
                );
            }
        }
    }

    /// THE KEY TEST: Two-body Hessian with pow_log produces accurate f32 results.
    ///
    /// Builds a_x = -mu * x / r³ using powf_log(r2, -1.5) instead of powf(r2, -1.5),
    /// then verifies that f32 simulation matches analytical Hessian within 1% tolerance
    /// (vs the >100% errors with standard pow).
    #[test]
    fn two_body_hessian_pow_log_f32_stable() {
        use crate::AutoDiff;

        let mut ad = AutoDiff::new();
        let rx = ad.var(0.0).unwrap();
        let ry = ad.var(0.0).unwrap();
        let rz = ad.var(0.0).unwrap();

        let rx2 = ad.mul(rx, rx);
        let ry2 = ad.mul(ry, ry);
        let rz2 = ad.mul(rz, rz);
        let rxy2 = ad.add(rx2, ry2);
        let r2 = ad.add(rxy2, rz2);

        // KEY DIFFERENCE: use powf_log instead of manual sqrt+mul+div
        let neg_mu = ad.constant(-398600.4418);
        let r2_neg1p5 = ad.powf_log(r2, -1.5); // r²^(-3/2) = r^(-3)
        let factor = ad.mul(neg_mu, r2_neg1p5);
        let accel = ad.mul(factor, rx); // a_x = -mu * x * r^(-3)

        let mut graph = ad.compile_order(accel, &[rx, ry, rz], 2).unwrap();

        let pos = [5000.0, 3000.0, 1000.0];
        graph.eval(&pos).unwrap();

        // f64 values (used for sanity checks below)
        let d2_0_0_f64 = graph.partial(&[2, 0, 0]).unwrap();
        let d1_1_0_f64 = graph.partial(&[1, 1, 0]).unwrap();
        let _d1_0_1_f64 = graph.partial(&[1, 0, 1]).unwrap();
        let _d0_2_0_f64 = graph.partial(&[0, 2, 0]).unwrap();
        let _d0_1_1_f64 = graph.partial(&[0, 1, 1]).unwrap();
        let _d0_0_2_f64 = graph.partial(&[0, 0, 2]).unwrap();

        // Analytical Hessian
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let mu = 398600.4418;
        let r2_val = x * x + y * y + z * z;
        let r = r2_val.sqrt();
        let r7 = r.powi(7);
        let d = |i: usize, j: usize| -> f64 { if i == j { 1.0 } else { 0.0 } };
        let ri = [x, y, z];
        let analytical_hessian = |j: usize, k: usize| -> f64 {
            (mu / r7)
                * (-15.0 * x * ri[j] * ri[k]
                    + 3.0 * r2_val * (ri[k] * d(0, j) + ri[j] * d(0, k) + x * d(j, k)))
        };

        let rel_err = |a: f64, b: f64| -> f64 {
            if b.abs() < 1e-30 { a.abs() } else { (a - b).abs() / b.abs() }
        };

        // Verify f64 eval matches analytical (sanity check)
        assert!(rel_err(d2_0_0_f64, analytical_hessian(0, 0)) < 1e-10,
            "f64 d2_0_0 doesn't match analytical");
        assert!(rel_err(d1_1_0_f64, analytical_hessian(0, 1)) < 1e-10,
            "f64 d1_1_0 doesn't match analytical");

        // --- f32 simulation: the actual test ---
        let nodes = graph.nodes();
        let partial_outputs = graph.partial_outputs();
        let pos_f32: Vec<f32> = pos.iter().map(|&v| v as f32).collect();
        let vals_f32 = eval_f32(nodes, &pos_f32);

        let find_partial = |mi: &[usize]| -> f32 {
            let idx = partial_outputs.iter().find(|(m, _)| m == mi).unwrap().1;
            vals_f32[idx]
        };

        let hessian_f32 = [
            (find_partial(&[2, 0, 0]), analytical_hessian(0, 0), "d2a/dx²"),
            (find_partial(&[1, 1, 0]), analytical_hessian(0, 1), "d2a/dxdy"),
            (find_partial(&[1, 0, 1]), analytical_hessian(0, 2), "d2a/dxdz"),
            (find_partial(&[0, 2, 0]), analytical_hessian(1, 1), "d2a/dy²"),
            (find_partial(&[0, 1, 1]), analytical_hessian(1, 2), "d2a/dydz"),
            (find_partial(&[0, 0, 2]), analytical_hessian(2, 2), "d2a/dz²"),
        ];

        eprintln!("\n=== pow_log two-body Hessian f32 stability ===");
        for &(f32_val, analytical, name) in &hessian_f32 {
            let err = rel_err(f32_val as f64, analytical);
            eprintln!("{name}: f32={f32_val:>15.6e}  analytical={analytical:>15.6e}  rel_err={err:.2e}");
            assert!(
                err < 0.01,
                "pow_log f32 {name} relative error {err:.4e} exceeds 1% — \
                 f32={f32_val}, analytical={analytical}"
            );
        }
    }

    /// div_log second-order: matches div second derivative.
    #[test]
    fn div_log_second_order_matches_div() {
        use crate::AutoDiff;

        // f = x / y, d²f/dxdy = -1/y²
        for &x_val in &[1.0, 2.0, 3.0] {
            for &y_val in &[0.5, 1.0, 2.0] {
                let mut ad_div = AutoDiff::new();
                let x1 = ad_div.var(x_val).unwrap();
                let y1 = ad_div.var(y_val).unwrap();
                let f1 = ad_div.div(x1, y1);
                let df1 = ad_div.differentiate(f1, x1).unwrap();
                let d2f1 = ad_div.differentiate(df1, y1).unwrap();
                let d2_div = ad_div.eval(d2f1).unwrap();

                let mut ad_log = AutoDiff::new();
                let x2 = ad_log.var(x_val).unwrap();
                let y2 = ad_log.var(y_val).unwrap();
                let f2 = ad_log.div_log(x2, y2);
                let df2 = ad_log.differentiate(f2, x2).unwrap();
                let d2f2 = ad_log.differentiate(df2, y2).unwrap();
                let d2_log = ad_log.eval(d2f2).unwrap();

                let expected = -1.0 / (y_val * y_val);
                assert!(
                    (d2_div - expected).abs() < 1e-10,
                    "div d²f/dxdy at x={x_val},y={y_val}: got {d2_div}, expected {expected}"
                );
                assert!(
                    (d2_log - expected).abs() < 1e-10,
                    "div_log d²f/dxdy at x={x_val},y={y_val}: got {d2_log}, expected {expected}"
                );
            }
        }
    }

    /// pow_log primal evaluation matches pow.
    #[test]
    fn pow_log_primal_matches_pow() {
        use crate::AutoDiff;

        for &(x_val, p) in &[(2.0, 3.0), (4.0, 0.5), (3.0, -2.0), (1.0, 10.0)] {
            let mut ad = AutoDiff::new();
            let x = ad.var(x_val).unwrap();
            let f_pow = ad.powf(x, p);
            let f_log = ad.powf_log(x, p);
            let v_pow = ad.eval(f_pow).unwrap();
            let v_log = ad.eval(f_log).unwrap();
            assert!(
                (v_pow - v_log).abs() < 1e-15,
                "primal mismatch at x={x_val},p={p}: pow={v_pow}, log={v_log}"
            );
        }
    }

    /// div_log primal evaluation matches div.
    #[test]
    fn div_log_primal_matches_div() {
        use crate::AutoDiff;

        for &(x_val, y_val) in &[(6.0, 3.0), (1.0, 7.0), (10.0, 0.1)] {
            let mut ad = AutoDiff::new();
            let x = ad.var(x_val).unwrap();
            let y = ad.var(y_val).unwrap();
            let f_div = ad.div(x, y);
            let f_log = ad.div_log(x, y);
            let v_div = ad.eval(f_div).unwrap();
            let v_log = ad.eval(f_log).unwrap();
            assert!(
                (v_div - v_log).abs() < 1e-15,
                "primal mismatch at x={x_val},y={y_val}: div={v_div}, log={v_log}"
            );
        }
    }
}
