//! Procedural macros for bevy_autodiff automatic differentiation.
//!
//! This crate provides the `#[autodiff]` attribute macro that transforms
//! functions to work with the bevy_autodiff autodiff system.
//!
//! # Example
//!
//! ```ignore
//! use bevy_autodiff::Var;
//! use bevy_autodiff_macros::autodiff;
//!
//! #[autodiff]
//! fn quadratic(x: Var) -> Var {
//!     x * x + 2.0 * x + 1.0
//! }
//!
//! // Expands to code that uses temporary variables to avoid borrow issues:
//! // fn quadratic(ad: &mut AutoDiff, x: Var) -> Var {
//! //     let __t0 = ad.mul(x, x);
//! //     let __t1 = ad.constant(2.0);
//! //     let __t2 = ad.mul(__t1, x);
//! //     let __t3 = ad.add(__t0, __t2);
//! //     let __t4 = ad.constant(1.0);
//! //     ad.add(__t3, __t4)
//! // }
//! ```

use proc_macro::TokenStream;
use quote::{format_ident, ToTokens};
use syn::{
    parse_macro_input, parse_quote, visit_mut::VisitMut, Expr, ExprBinary, ExprCall, ExprLit,
    ExprMethodCall, ExprPath, ExprUnary, FnArg, ItemFn, Lit, Pat, Stmt, UnOp,
};

/// Transforms a function to work with AutoDiff.
///
/// This attribute macro:
/// - Adds `ad: &mut AutoDiff` as the first parameter
/// - Transforms arithmetic operators (`+`, `-`, `*`, `/`) to AutoDiff method calls
/// - Transforms math functions (`sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `sinh`, `cosh`, `tanh`, etc.) to method calls
/// - Wraps float literals as `ad.constant(value)`
/// - Uses temporary variables to avoid borrow checker issues
///
/// # Example
///
/// ```ignore
/// use bevy_autodiff::{AutoDiff, Var};
/// use bevy_autodiff_macros::autodiff;
///
/// #[autodiff]
/// fn rosenbrock(x: Var, y: Var) -> Var {
///     let a = 1.0;
///     let b = 100.0;
///     (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
/// }
///
/// // Usage:
/// let mut ad = AutoDiff::new();
/// let x = ad.var(1.0);
/// let y = ad.var(1.0);
/// let f = rosenbrock(&mut ad, x, y);
/// ```
///
/// # Supported Operations
///
/// - Binary: `+`, `-`, `*`, `/`
/// - Unary: `-` (negation)
/// - Functions: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`, `pow`, `powi`, `powf`, `square`, `pow_log`, `powi_log`, `powf_log`, `div_log`
/// - Method syntax: `x.sin()`, `x.cos()`, etc. — transformed to `ad.sin(x)`, `ad.cos(x)`, etc.
/// - Literals: float and integer literals become `ad.constant(value)`
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
/// - Variables bound with `let` that are floats should be used directly, not as Var
/// - The function must have `Var` parameters and return `Var`
#[proc_macro_attribute]
pub fn autodiff(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut func = parse_macro_input!(item as ItemFn);

    // Parse attribute for stable_derivatives flag
    let attr_str = attr.to_string();
    let stable_derivatives = attr_str.contains("stable_derivatives");

    // Add `ad: &mut AutoDiff` as first parameter
    let ad_param: FnArg = parse_quote!(ad: &mut AutoDiff);
    func.sig.inputs.insert(0, ad_param);

    // Transform the function body
    let mut transformer = ExprTransformer::new();
    transformer.stable_derivatives = stable_derivatives;
    transformer.visit_item_fn_mut(&mut func);

    TokenStream::from(func.into_token_stream())
}

/// Visitor that transforms expressions for autodiff.
struct ExprTransformer {
    /// Names of local variables that are NOT Var types (regular floats)
    local_float_vars: Vec<String>,
    /// Counter for generating unique temporary variable names
    temp_counter: usize,
    /// When true, routes pow → pow_log, div → div_log for f32-stable derivatives
    stable_derivatives: bool,
}

impl ExprTransformer {
    fn new() -> Self {
        Self {
            local_float_vars: Vec::new(),
            temp_counter: 0,
            stable_derivatives: false,
        }
    }

    /// Generate a unique temporary variable name
    fn next_temp(&mut self) -> syn::Ident {
        let name = format_ident!("__autodiff_tmp_{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

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

    /// Check if an identifier is a local float variable
    fn is_local_float(&self, name: &str) -> bool {
        self.local_float_vars.contains(&name.to_string())
    }

    /// Transform an expression, returning a block that evaluates to the result.
    /// This method handles creating temporary variables for nested expressions.
    fn transform_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Binary(binary) => self.transform_binary(binary),
            Expr::Unary(unary) => self.transform_unary(unary),
            Expr::Call(call) => self.transform_call(call),
            Expr::MethodCall(method_call) => self.transform_method_call(method_call),
            Expr::Lit(lit) => self.transform_literal(lit),
            Expr::Path(path) => self.transform_path(path),
            Expr::Paren(paren) => self.transform_expr(&paren.expr),
            _ => expr.clone(),
        }
    }

    /// Transform a binary expression to a block with temp variables
    fn transform_binary(&mut self, expr: &ExprBinary) -> Expr {
        let method_name = match &expr.op {
            syn::BinOp::Add(_) => "add",
            syn::BinOp::Sub(_) => "sub",
            syn::BinOp::Mul(_) => "mul",
            syn::BinOp::Div(_) => {
                if self.stable_derivatives { "div_log" } else { "div" }
            }
            _ => return Expr::Binary(expr.clone()), // Don't transform other operators
        };

        // Transform operands
        let left_transformed = self.transform_expr(&expr.left);
        let right_transformed = self.transform_expr(&expr.right);

        // Create temp variables and the method call
        let lhs_temp = self.next_temp();
        let rhs_temp = self.next_temp();
        let method = syn::Ident::new(method_name, proc_macro2::Span::call_site());

        parse_quote!({
            let #lhs_temp = #left_transformed;
            let #rhs_temp = #right_transformed;
            ad.#method(#lhs_temp, #rhs_temp)
        })
    }

    /// Transform a unary expression (negation)
    fn transform_unary(&mut self, expr: &ExprUnary) -> Expr {
        match &expr.op {
            UnOp::Neg(_) => {
                let operand_transformed = self.transform_expr(&expr.expr);
                let temp = self.next_temp();

                parse_quote!({
                    let #temp = #operand_transformed;
                    ad.neg(#temp)
                })
            }
            _ => Expr::Unary(expr.clone()),
        }
    }

    /// Transform a function call (for math functions)
    fn transform_call(&mut self, expr: &ExprCall) -> Expr {
        // Get the function name
        let func_name = if let Expr::Path(path) = &*expr.func {
            path.path.get_ident().map(|id| id.to_string())
        } else {
            None
        };

        let func_name = match func_name {
            Some(name) if Self::is_math_function(&name) => name,
            _ => {
                // Not a math function we transform, but transform arguments
                let mut new_call = expr.clone();
                for arg in &mut new_call.args {
                    *arg = self.transform_expr(arg);
                }
                return Expr::Call(new_call);
            }
        };

        // Handle pow / pow_log (two Var arguments)
        if (func_name == "pow" || func_name == "pow_log") && expr.args.len() == 2 {
            let base = self.transform_expr(&expr.args[0]);
            let exp = self.transform_expr(&expr.args[1]);
            let base_temp = self.next_temp();
            let exp_temp = self.next_temp();

            let method_name = if func_name == "pow_log" || self.stable_derivatives {
                "pow_log"
            } else {
                "pow"
            };
            let method = syn::Ident::new(method_name, proc_macro2::Span::call_site());

            return parse_quote!({
                let #base_temp = #base;
                let #exp_temp = #exp;
                ad.#method(#base_temp, #exp_temp)
            });
        }

        // Handle powi / powi_log (Var + raw i32)
        if (func_name == "powi" || func_name == "powi_log") && expr.args.len() == 2 {
            let base = self.transform_expr(&expr.args[0]);
            let base_temp = self.next_temp();
            let exp_arg = &expr.args[1]; // raw i32, not transformed

            let method_name = if func_name == "powi_log" || self.stable_derivatives {
                "powi_log"
            } else {
                "powi"
            };
            let method = syn::Ident::new(method_name, proc_macro2::Span::call_site());

            return parse_quote!({
                let #base_temp = #base;
                ad.#method(#base_temp, #exp_arg)
            });
        }

        // Handle powf / powf_log (Var + raw f64)
        if (func_name == "powf" || func_name == "powf_log") && expr.args.len() == 2 {
            let base = self.transform_expr(&expr.args[0]);
            let base_temp = self.next_temp();
            let exp_arg = &expr.args[1]; // raw f64, not transformed

            let method_name = if func_name == "powf_log" || self.stable_derivatives {
                "powf_log"
            } else {
                "powf"
            };
            let method = syn::Ident::new(method_name, proc_macro2::Span::call_site());

            return parse_quote!({
                let #base_temp = #base;
                ad.#method(#base_temp, #exp_arg)
            });
        }

        // Handle div_log (two Var arguments)
        if func_name == "div_log" && expr.args.len() == 2 {
            let lhs = self.transform_expr(&expr.args[0]);
            let rhs = self.transform_expr(&expr.args[1]);
            let lhs_temp = self.next_temp();
            let rhs_temp = self.next_temp();

            return parse_quote!({
                let #lhs_temp = #lhs;
                let #rhs_temp = #rhs;
                ad.div_log(#lhs_temp, #rhs_temp)
            });
        }

        // Single argument functions (sin, cos, ..., square)
        if expr.args.len() != 1 {
            return Expr::Call(expr.clone());
        }

        let arg = self.transform_expr(&expr.args[0]);
        let temp = self.next_temp();
        let method = syn::Ident::new(&func_name, proc_macro2::Span::call_site());

        parse_quote!({
            let #temp = #arg;
            ad.#method(#temp)
        })
    }

    /// Transform method-call syntax: `x.sin()` → `ad.sin(x)`, `x.powi(3)` → `ad.powi(x, 3)`
    fn transform_method_call(&mut self, expr: &ExprMethodCall) -> Expr {
        let method_name = expr.method.to_string();

        if !Self::is_math_function(&method_name) {
            // Not a math method — transform receiver and args, pass through
            let mut new_expr = expr.clone();
            new_expr.receiver = Box::new(self.transform_expr(&expr.receiver));
            for arg in &mut new_expr.args {
                *arg = self.transform_expr(arg);
            }
            return Expr::MethodCall(new_expr);
        }

        let receiver = self.transform_expr(&expr.receiver);
        let recv_temp = self.next_temp();
        let method = syn::Ident::new(&method_name, proc_macro2::Span::call_site());

        // powi / powi_log: x.powi(3) → ad.powi(x, 3) — second arg is raw integer
        if (method_name == "powi" || method_name == "powi_log") && expr.args.len() == 1 {
            let exp_arg = &expr.args[0]; // raw i32, not transformed
            let target_method = if method_name == "powi_log" || self.stable_derivatives {
                "powi_log"
            } else {
                "powi"
            };
            let target = syn::Ident::new(target_method, proc_macro2::Span::call_site());
            return parse_quote!({
                let #recv_temp = #receiver;
                ad.#target(#recv_temp, #exp_arg)
            });
        }

        // powf / powf_log: x.powf(2.5) → ad.powf(x, 2.5) — second arg is raw f64
        if (method_name == "powf" || method_name == "powf_log") && expr.args.len() == 1 {
            let exp_arg = &expr.args[0]; // raw f64, not transformed
            let target_method = if method_name == "powf_log" || self.stable_derivatives {
                "powf_log"
            } else {
                "powf"
            };
            let target = syn::Ident::new(target_method, proc_macro2::Span::call_site());
            return parse_quote!({
                let #recv_temp = #receiver;
                ad.#target(#recv_temp, #exp_arg)
            });
        }

        // Zero-arg method calls: x.sin(), x.cos(), x.square(), etc.
        if expr.args.is_empty() {
            return parse_quote!({
                let #recv_temp = #receiver;
                ad.#method(#recv_temp)
            });
        }

        // Fallback: not a recognized pattern, pass through
        Expr::MethodCall(expr.clone())
    }

    /// Transform a literal to ad.constant(value)
    fn transform_literal(&mut self, expr: &ExprLit) -> Expr {
        match &expr.lit {
            Lit::Float(f) => {
                if let Ok(value) = f.base10_parse::<f64>() {
                    parse_quote!(ad.constant(#value))
                } else {
                    Expr::Lit(expr.clone())
                }
            }
            Lit::Int(i) => {
                if let Ok(value) = i.base10_parse::<i64>() {
                    let float_value = value as f64;
                    parse_quote!(ad.constant(#float_value))
                } else {
                    Expr::Lit(expr.clone())
                }
            }
            _ => Expr::Lit(expr.clone()),
        }
    }

    /// Transform a path expression (variable reference)
    fn transform_path(&mut self, expr: &ExprPath) -> Expr {
        if let Some(ident) = expr.path.get_ident() {
            if self.is_local_float(&ident.to_string()) {
                // Wrap local float variables with ad.constant()
                return parse_quote!(ad.constant(#ident));
            }
        }
        Expr::Path(expr.clone())
    }
}

impl VisitMut for ExprTransformer {
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Local(local) => {
                // Track local variables that are assigned float literals
                if let Some(init) = &local.init {
                    if matches!(
                        &*init.expr,
                        Expr::Lit(ExprLit {
                            lit: Lit::Float(_),
                            ..
                        }) | Expr::Lit(ExprLit {
                            lit: Lit::Int(_),
                            ..
                        })
                    ) {
                        if let Pat::Ident(pat_ident) = &local.pat {
                            self.local_float_vars.push(pat_ident.ident.to_string());
                            // Don't transform this initialization - it's a regular float
                            return;
                        }
                    }
                }

                // Transform the initialization expression
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
