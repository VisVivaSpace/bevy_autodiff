//! Procedural macros for bevy_autodiff automatic differentiation.
//!
//! This crate provides the `#[autodiff]` attribute macro that transforms
//! functions to be generic over [`DiffNum`](https://docs.rs/bevy_autodiff/latest/bevy_autodiff/trait.DiffNum.html),
//! enabling dual-use as both direct float computation and automatic
//! differentiation graph construction.
//!
//! # Example
//!
//! ```ignore
//! use bevy_autodiff::{autodiff, DiffNum};
//!
//! #[autodiff]
//! fn quadratic(x: f64) -> f64 {
//!     x * x + 2.0 * x + 1.0
//! }
//!
//! // Direct float evaluation:
//! assert_eq!(quadratic(3.0_f64), 16.0);
//!
//! // AD graph construction:
//! use bevy_autodiff::{AutoDiff, Var, ops::with_context};
//! let mut ad = AutoDiff::new();
//! let x = ad.var(3.0).unwrap();
//! let f = with_context(&mut ad, || quadratic(x));
//! assert_eq!(ad.eval(f).unwrap(), 16.0);
//! ```

use proc_macro::TokenStream;
use quote::ToTokens;
use syn::{
    parse_macro_input, parse_quote, visit_mut::VisitMut, Expr, ExprBinary, ExprCall, ExprLit,
    ExprMethodCall, ExprUnary, FnArg, ItemFn, Lit, Stmt, Type,
};

/// Transforms a function to be generic over `DiffNum`.
///
/// This attribute macro:
/// - Replaces parameter types (`f64`, `f32`, `Var`) with a generic `T: DiffNum`
/// - Transforms float/integer literals to `T::from_f64(value)`
/// - Transforms free math function calls (`sin(x)`) to method calls (`(x).sin()`)
/// - Leaves method calls (`.sin()`, `.cos()`, etc.) as-is (resolved via `DiffNum` bound)
/// - Leaves arithmetic operators (`+`, `-`, `*`, `/`) as-is (resolved via `DiffNum` supertraits)
///
/// # Example
///
/// ```ignore
/// use bevy_autodiff::{autodiff, DiffNum};
///
/// #[autodiff]
/// fn rosenbrock(x: f64, y: f64) -> f64 {
///     let a = 1.0;
///     let b = 100.0;
///     (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
/// }
///
/// // Direct evaluation with f64:
/// assert_eq!(rosenbrock(1.0_f64, 1.0), 0.0);
///
/// // AD graph construction with Var:
/// use bevy_autodiff::{AutoDiff, ops::with_context};
/// let mut ad = AutoDiff::new();
/// let x = ad.var(1.0).unwrap();
/// let y = ad.var(1.0).unwrap();
/// let f = with_context(&mut ad, || rosenbrock(x, y));
/// ```
///
/// # Supported Operations
///
/// - Binary: `+`, `-`, `*`, `/` (via `DiffNum` supertraits)
/// - Unary: `-` (negation, via `Neg` supertrait)
/// - Functions: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `sinh`, `cosh`, `tanh`,
///   `asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`, `pow`, `powi`, `powf`,
///   `square`, `pow_log`, `powi_log`, `powf_log`, `div_log`
/// - Method syntax: `x.sin()`, `x.powi(3)`, etc.
/// - Literals: float and integer literals become `T::from_f64(value)`
///
/// # `stable_derivatives` Attribute
///
/// Use `#[autodiff(stable_derivatives)]` to automatically route `pow` → `pow_log`,
/// `powi` → `powi_log`, `powf` → `powf_log`, and `/` → `div_log`. This produces
/// derivative graphs that avoid catastrophic cancellation in f32 for second-order
/// and higher derivatives.
///
/// **Requirement:** all bases must be positive and all divisors nonzero.
///
/// # Limitations
///
/// - Only transforms expressions, not control flow (if/else, loops)
/// - `powi`/`powi_log` second argument must be a literal integer, not a variable
/// - `DiffNum` must be in scope at the definition site (e.g., `use bevy_autodiff::DiffNum;`)
#[proc_macro_attribute]
pub fn autodiff(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut func = parse_macro_input!(item as ItemFn);

    // Parse attribute for stable_derivatives flag
    let stable_derivatives = if attr.is_empty() {
        false
    } else {
        let ident: syn::Ident = match syn::parse(attr) {
            Ok(ident) => ident,
            Err(e) => return syn::Error::new(e.span(), "expected `stable_derivatives`").to_compile_error().into(),
        };
        if ident != "stable_derivatives" {
            return syn::Error::new(ident.span(), format!("unknown autodiff attribute: `{ident}`. Expected `stable_derivatives`.")).to_compile_error().into();
        }
        true
    };

    // Add generic parameter T: DiffNum
    let type_param: syn::GenericParam = parse_quote!(T: DiffNum);
    func.sig.generics.params.push(type_param);

    // Replace parameter types (f64, f32, Var) with T
    for param in &mut func.sig.inputs {
        if let FnArg::Typed(pat_type) = param {
            if is_autodiff_type(&pat_type.ty) {
                pat_type.ty = Box::new(parse_quote!(T));
            }
        }
    }

    // Replace return type
    if let syn::ReturnType::Type(_, ref mut ty) = func.sig.output {
        if is_autodiff_type(ty) {
            **ty = parse_quote!(T);
        }
    }

    // Transform the function body
    let mut transformer = DiffNumTransformer { stable_derivatives };
    transformer.visit_item_fn_mut(&mut func);

    TokenStream::from(func.into_token_stream())
}

/// Check if a type is one we should replace with T.
fn is_autodiff_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(ident) = type_path.path.get_ident() {
            let name = ident.to_string();
            return matches!(name.as_str(), "f64" | "f32" | "Var");
        }
    }
    false
}

/// Visitor that transforms expressions for DiffNum-generic functions.
///
/// Key transformations:
/// - Float/int literals → `T::from_f64(value)`
/// - Free function calls (`sin(x)`) → method calls (`(x).sin()`)
/// - Method calls stay as-is (resolved via `DiffNum` bound on `T`)
/// - Binary/unary operators stay as-is (resolved via `DiffNum` supertraits)
/// - With `stable_derivatives`: `/` → `.div_log()`, pow/powi/powf → log variants
struct DiffNumTransformer {
    /// When true, routes pow → pow_log, div → div_log for f32-stable derivatives
    stable_derivatives: bool,
}

impl DiffNumTransformer {
    /// Check if an identifier is a known math function
    fn is_math_function(name: &str) -> bool {
        matches!(
            name,
            "sin" | "cos" | "tan" | "exp" | "ln" | "sqrt"
            | "sinh" | "cosh" | "tanh"
            | "asin" | "acos" | "atan"
            | "asinh" | "acosh" | "atanh"
            | "pow" | "powi" | "powf" | "square"
            | "pow_log" | "powi_log" | "powf_log" | "div_log"
        )
    }

    /// Recursively transform an expression.
    fn transform_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Binary(binary) => self.transform_binary(binary),
            Expr::Unary(unary) => self.transform_unary(unary),
            Expr::Call(call) => self.transform_call(call),
            Expr::MethodCall(mc) => self.transform_method_call(mc),
            Expr::Lit(lit) => self.transform_literal(lit),
            Expr::Paren(paren) => {
                let inner = self.transform_expr(&paren.expr);
                parse_quote!((#inner))
            }
            _ => expr.clone(),
        }
    }

    /// Transform a literal to `T::from_f64(value)`.
    fn transform_literal(&mut self, lit: &ExprLit) -> Expr {
        match &lit.lit {
            Lit::Float(f) => {
                if let Ok(value) = f.base10_parse::<f64>() {
                    parse_quote!(T::from_f64(#value))
                } else {
                    Expr::Lit(lit.clone())
                }
            }
            Lit::Int(i) => {
                if let Ok(value) = i.base10_parse::<i64>() {
                    let float_value = value as f64;
                    parse_quote!(T::from_f64(#float_value))
                } else {
                    Expr::Lit(lit.clone())
                }
            }
            _ => Expr::Lit(lit.clone()),
        }
    }

    /// Transform a binary expression — operators stay as-is, except `/` with `stable_derivatives`.
    fn transform_binary(&mut self, expr: &ExprBinary) -> Expr {
        let left = self.transform_expr(&expr.left);
        let right = self.transform_expr(&expr.right);

        // stable_derivatives: route / to div_log
        if self.stable_derivatives {
            if let syn::BinOp::Div(_) = &expr.op {
                return parse_quote!((#left).div_log(#right));
            }
        }

        // Reconstruct with transformed operands, preserving the operator
        Expr::Binary(ExprBinary {
            attrs: expr.attrs.clone(),
            left: Box::new(left),
            op: expr.op.clone(),
            right: Box::new(right),
        })
    }

    /// Transform a unary expression — operators stay as-is.
    fn transform_unary(&mut self, expr: &ExprUnary) -> Expr {
        let operand = self.transform_expr(&expr.expr);
        Expr::Unary(ExprUnary {
            attrs: expr.attrs.clone(),
            op: expr.op.clone(),
            expr: Box::new(operand),
        })
    }

    /// Transform a free function call to a method call on the first argument.
    ///
    /// `sin(x)` → `(x).sin()`, `pow(x, y)` → `(x).powf(y)`, etc.
    fn transform_call(&mut self, call: &ExprCall) -> Expr {
        let func_name = if let Expr::Path(path) = &*call.func {
            path.path.get_ident().map(|id| id.to_string())
        } else {
            None
        };

        let func_name = match func_name {
            Some(name) if Self::is_math_function(&name) => name,
            _ => {
                // Not a math function — transform arguments, pass through
                let mut new_call = call.clone();
                for arg in &mut new_call.args {
                    *arg = self.transform_expr(arg);
                }
                return Expr::Call(new_call);
            }
        };

        match func_name.as_str() {
            // Single-arg: sin(x) → (x).sin()
            "sin" | "cos" | "tan" | "exp" | "ln" | "sqrt" | "square"
            | "sinh" | "cosh" | "tanh"
            | "asin" | "acos" | "atan"
            | "asinh" | "acosh" | "atanh" => {
                if call.args.len() == 1 {
                    let arg = self.transform_expr(&call.args[0]);
                    let method = syn::Ident::new(&func_name, proc_macro2::Span::call_site());
                    parse_quote!((#arg).#method())
                } else {
                    Expr::Call(call.clone())
                }
            }

            // pow(x, y) → (x).powf(y) or (x).pow_log(y)
            "pow" | "pow_log" => {
                if call.args.len() == 2 {
                    let base = self.transform_expr(&call.args[0]);
                    let exp = self.transform_expr(&call.args[1]);
                    if func_name == "pow_log" || self.stable_derivatives {
                        parse_quote!((#base).pow_log(#exp))
                    } else {
                        parse_quote!((#base).powf(#exp))
                    }
                } else {
                    Expr::Call(call.clone())
                }
            }

            // powi(x, 3) → (x).powi(3) — second arg is raw i32, NOT transformed
            "powi" | "powi_log" => {
                if call.args.len() == 2 {
                    let base = self.transform_expr(&call.args[0]);
                    let n = &call.args[1]; // raw i32
                    if func_name == "powi_log" || self.stable_derivatives {
                        parse_quote!((#base).powi_log(#n))
                    } else {
                        parse_quote!((#base).powi(#n))
                    }
                } else {
                    Expr::Call(call.clone())
                }
            }

            // powf(x, 0.5) → (x).powf(T::from_f64(0.5))
            "powf" | "powf_log" => {
                if call.args.len() == 2 {
                    let base = self.transform_expr(&call.args[0]);
                    let exp = self.transform_expr(&call.args[1]); // transforms literal
                    if func_name == "powf_log" || self.stable_derivatives {
                        parse_quote!((#base).powf_log(#exp))
                    } else {
                        parse_quote!((#base).powf(#exp))
                    }
                } else {
                    Expr::Call(call.clone())
                }
            }

            // div_log(x, y) → (x).div_log(y)
            "div_log" => {
                if call.args.len() == 2 {
                    let lhs = self.transform_expr(&call.args[0]);
                    let rhs = self.transform_expr(&call.args[1]);
                    parse_quote!((#lhs).div_log(#rhs))
                } else {
                    Expr::Call(call.clone())
                }
            }

            _ => Expr::Call(call.clone()),
        }
    }

    /// Transform a method call — mostly left as-is since `DiffNum` methods resolve via bound.
    ///
    /// With `stable_derivatives`, routes powi/powf/pow to log variants.
    fn transform_method_call(&mut self, mc: &ExprMethodCall) -> Expr {
        let method_name = mc.method.to_string();

        if !Self::is_math_function(&method_name) {
            // Not a math method — transform receiver and args, pass through
            let mut new_mc = mc.clone();
            new_mc.receiver = Box::new(self.transform_expr(&mc.receiver));
            for arg in &mut new_mc.args {
                *arg = self.transform_expr(arg);
            }
            return Expr::MethodCall(new_mc);
        }

        let receiver = self.transform_expr(&mc.receiver);

        match method_name.as_str() {
            // Zero-arg methods: x.sin() — stay as method call with transformed receiver
            "sin" | "cos" | "tan" | "exp" | "ln" | "sqrt" | "square"
            | "sinh" | "cosh" | "tanh"
            | "asin" | "acos" | "atan"
            | "asinh" | "acosh" | "atanh" if mc.args.is_empty() => {
                let method = &mc.method;
                parse_quote!((#receiver).#method())
            }

            // x.powi(n) — second arg is raw i32, not transformed
            "powi" | "powi_log" if mc.args.len() == 1 => {
                let n = &mc.args[0]; // raw i32
                if self.stable_derivatives || method_name == "powi_log" {
                    parse_quote!((#receiver).powi_log(#n))
                } else {
                    parse_quote!((#receiver).powi(#n))
                }
            }

            // x.powf(e) — second arg is transformed (becomes T)
            "powf" | "powf_log" if mc.args.len() == 1 => {
                let exp = self.transform_expr(&mc.args[0]);
                if self.stable_derivatives || method_name == "powf_log" {
                    parse_quote!((#receiver).powf_log(#exp))
                } else {
                    parse_quote!((#receiver).powf(#exp))
                }
            }

            // x.pow(y) or x.pow_log(y)
            "pow" | "pow_log" if mc.args.len() == 1 => {
                let exp = self.transform_expr(&mc.args[0]);
                if self.stable_derivatives || method_name == "pow_log" {
                    parse_quote!((#receiver).pow_log(#exp))
                } else {
                    parse_quote!((#receiver).powf(#exp))
                }
            }

            // x.div_log(y)
            "div_log" if mc.args.len() == 1 => {
                let rhs = self.transform_expr(&mc.args[0]);
                parse_quote!((#receiver).div_log(#rhs))
            }

            _ => {
                let mut new_mc = mc.clone();
                new_mc.receiver = Box::new(receiver);
                for arg in &mut new_mc.args {
                    *arg = self.transform_expr(arg);
                }
                Expr::MethodCall(new_mc)
            }
        }
    }
}

impl VisitMut for DiffNumTransformer {
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Local(local) => {
                // Transform the initialization expression (float literals → T::from_f64)
                if let Some(init) = &mut local.init {
                    let transformed = self.transform_expr(&init.expr);
                    init.expr = Box::new(transformed);
                }
            }
            Stmt::Expr(expr, _) => {
                let transformed = self.transform_expr(expr);
                *expr = transformed;
            }
            _ => {
                syn::visit_mut::visit_stmt_mut(self, stmt);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Proc-macro crates can't have unit tests that use the macro directly
    // Tests are in the main bevy_autodiff crate
}
