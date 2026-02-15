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
        BinaryOp::Div => format!("{lhs} / {rhs}"),
        BinaryOp::Pow => format!("pow({lhs}, {rhs})"),
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
}
