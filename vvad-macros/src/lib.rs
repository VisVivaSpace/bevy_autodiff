//! Procedural macros for vvad automatic differentiation.
//!
//! This crate provides the `#[autodiff]` attribute macro that transforms
//! functions to work with the vvad autodiff system.
//!
//! # Example
//!
//! ```ignore
//! use vvad::Var;
//! use vvad_macros::autodiff;
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
    ExprPath, ExprUnary, FnArg, ItemFn, Lit, Pat, Stmt, UnOp,
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
/// use vvad::{AutoDiff, Var};
/// use vvad_macros::autodiff;
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
/// - Functions: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`, `pow`
/// - Literals: float and integer literals become `ad.constant(value)`
///
/// # Limitations
///
/// - Only transforms expressions, not control flow (if/else, loops)
/// - Variables bound with `let` that are floats should be used directly, not as Var
/// - The function must have `Var` parameters and return `Var`
#[proc_macro_attribute]
pub fn autodiff(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut func = parse_macro_input!(item as ItemFn);

    // Add `ad: &mut AutoDiff` as first parameter
    let ad_param: FnArg = parse_quote!(ad: &mut AutoDiff);
    func.sig.inputs.insert(0, ad_param);

    // Transform the function body
    let mut transformer = ExprTransformer::new();
    transformer.visit_item_fn_mut(&mut func);

    TokenStream::from(func.into_token_stream())
}

/// Visitor that transforms expressions for autodiff.
struct ExprTransformer {
    /// Names of local variables that are NOT Var types (regular floats)
    local_float_vars: Vec<String>,
    /// Counter for generating unique temporary variable names
    temp_counter: usize,
}

impl ExprTransformer {
    fn new() -> Self {
        Self {
            local_float_vars: Vec::new(),
            temp_counter: 0,
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
            | "pow"
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
            syn::BinOp::Div(_) => "div",
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

        // Handle pow specially (two arguments)
        if func_name == "pow" && expr.args.len() == 2 {
            let base = self.transform_expr(&expr.args[0]);
            let exp = self.transform_expr(&expr.args[1]);
            let base_temp = self.next_temp();
            let exp_temp = self.next_temp();

            return parse_quote!({
                let #base_temp = #base;
                let #exp_temp = #exp;
                ad.pow(#base_temp, #exp_temp)
            });
        }

        // Single argument functions
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
    // Tests are in the main vvad crate
}
