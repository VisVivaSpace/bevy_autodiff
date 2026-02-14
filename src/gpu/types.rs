//! GPU-friendly data types and conversion from CPU representations.

use bytemuck::{Pod, Zeroable};

use crate::compiled::NodeOp;
use crate::components::{BinaryOp, UnaryOp};

/// GPU-friendly packed node representation (32 bytes, aligned).
///
/// Mirrors [`NodeOp`](crate::NodeOp) but with fixed-size fields suitable
/// for GPU storage buffers. Constants are stored as f32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuNodeOp {
    /// 0 = Input, 1 = Constant, 2 = Unary, 3 = Binary.
    pub op_type: u32,
    /// For Input: the input index. For Unary/Binary: the operation discriminant.
    pub op_code: u32,
    /// For Unary: source node index. For Binary: left operand node index.
    pub arg1: u32,
    /// For Binary: right operand node index. Unused otherwise.
    pub arg2: u32,
    /// For Constant: the value (f64 truncated to f32). Unused otherwise.
    pub const_val: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Op type discriminants (must match shader.wgsl)
const OP_TYPE_INPUT: u32 = 0;
const OP_TYPE_CONSTANT: u32 = 1;
const OP_TYPE_UNARY: u32 = 2;
const OP_TYPE_BINARY: u32 = 3;

/// Maps a [`UnaryOp`] to its GPU op code (must match shader.wgsl `eval_unary`).
pub(crate) fn unary_op_code(op: UnaryOp) -> u32 {
    match op {
        UnaryOp::Neg => 0,
        UnaryOp::Sin => 1,
        UnaryOp::Cos => 2,
        UnaryOp::Tan => 3,
        UnaryOp::Exp => 4,
        UnaryOp::Ln => 5,
        UnaryOp::Sqrt => 6,
        UnaryOp::Sinh => 7,
        UnaryOp::Cosh => 8,
        UnaryOp::Tanh => 9,
        UnaryOp::Asin => 10,
        UnaryOp::Acos => 11,
        UnaryOp::Atan => 12,
        UnaryOp::Asinh => 13,
        UnaryOp::Acosh => 14,
        UnaryOp::Atanh => 15,
    }
}

/// Maps a [`BinaryOp`] to its GPU op code (must match shader.wgsl `eval_binary`).
pub(crate) fn binary_op_code(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Pow => 4,
    }
}

/// Converts a slice of [`NodeOp`] to GPU-friendly [`GpuNodeOp`] representation.
pub(crate) fn convert_nodes(nodes: &[NodeOp]) -> Vec<GpuNodeOp> {
    nodes
        .iter()
        .map(|node| match *node {
            NodeOp::Input(idx) => GpuNodeOp {
                op_type: OP_TYPE_INPUT,
                op_code: idx as u32,
                arg1: 0,
                arg2: 0,
                const_val: 0.0,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            NodeOp::Constant(v) => GpuNodeOp {
                op_type: OP_TYPE_CONSTANT,
                op_code: 0,
                arg1: 0,
                arg2: 0,
                const_val: v as f32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            NodeOp::Unary { op, src } => GpuNodeOp {
                op_type: OP_TYPE_UNARY,
                op_code: unary_op_code(op),
                arg1: src as u32,
                arg2: 0,
                const_val: 0.0,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            NodeOp::Binary { op, lhs, rhs } => GpuNodeOp {
                op_type: OP_TYPE_BINARY,
                op_code: binary_op_code(op),
                arg1: lhs as u32,
                arg2: rhs as u32,
                const_val: 0.0,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_input_node() {
        let nodes = [NodeOp::Input(3)];
        let gpu = convert_nodes(&nodes);
        assert_eq!(gpu.len(), 1);
        assert_eq!(gpu[0].op_type, OP_TYPE_INPUT);
        assert_eq!(gpu[0].op_code, 3);
    }

    #[test]
    fn convert_constant_node() {
        let nodes = [NodeOp::Constant(3.14159265358979)];
        let gpu = convert_nodes(&nodes);
        assert_eq!(gpu[0].op_type, OP_TYPE_CONSTANT);
        assert!((gpu[0].const_val - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn convert_all_unary_ops() {
        let ops = [
            UnaryOp::Neg,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Tan,
            UnaryOp::Exp,
            UnaryOp::Ln,
            UnaryOp::Sqrt,
            UnaryOp::Sinh,
            UnaryOp::Cosh,
            UnaryOp::Tanh,
            UnaryOp::Asin,
            UnaryOp::Acos,
            UnaryOp::Atan,
            UnaryOp::Asinh,
            UnaryOp::Acosh,
            UnaryOp::Atanh,
        ];
        for (expected_code, &op) in ops.iter().enumerate() {
            assert_eq!(unary_op_code(op), expected_code as u32);
            let nodes = [NodeOp::Unary { op, src: 0 }];
            let gpu = convert_nodes(&nodes);
            assert_eq!(gpu[0].op_type, OP_TYPE_UNARY);
            assert_eq!(gpu[0].op_code, expected_code as u32);
            assert_eq!(gpu[0].arg1, 0);
        }
    }

    #[test]
    fn convert_all_binary_ops() {
        let ops = [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
        ];
        for (expected_code, &op) in ops.iter().enumerate() {
            assert_eq!(binary_op_code(op), expected_code as u32);
            let nodes = [NodeOp::Binary { op, lhs: 1, rhs: 2 }];
            let gpu = convert_nodes(&nodes);
            assert_eq!(gpu[0].op_type, OP_TYPE_BINARY);
            assert_eq!(gpu[0].op_code, expected_code as u32);
            assert_eq!(gpu[0].arg1, 1);
            assert_eq!(gpu[0].arg2, 2);
        }
    }

    #[test]
    fn convert_mixed_graph() {
        // f(x) = 2*x + 1: Input(0), Constant(2), Mul(1,0), Constant(1), Add(2,3)
        let nodes = vec![
            NodeOp::Input(0),
            NodeOp::Constant(2.0),
            NodeOp::Binary {
                op: BinaryOp::Mul,
                lhs: 1,
                rhs: 0,
            },
            NodeOp::Constant(1.0),
            NodeOp::Binary {
                op: BinaryOp::Add,
                lhs: 2,
                rhs: 3,
            },
        ];
        let gpu = convert_nodes(&nodes);
        assert_eq!(gpu.len(), 5);
        assert_eq!(gpu[0].op_type, OP_TYPE_INPUT);
        assert_eq!(gpu[1].op_type, OP_TYPE_CONSTANT);
        assert_eq!(gpu[1].const_val, 2.0);
        assert_eq!(gpu[2].op_type, OP_TYPE_BINARY);
        assert_eq!(gpu[2].op_code, binary_op_code(BinaryOp::Mul));
        assert_eq!(gpu[2].arg1, 1);
        assert_eq!(gpu[2].arg2, 0);
        assert_eq!(gpu[3].op_type, OP_TYPE_CONSTANT);
        assert_eq!(gpu[3].const_val, 1.0);
        assert_eq!(gpu[4].op_type, OP_TYPE_BINARY);
        assert_eq!(gpu[4].op_code, binary_op_code(BinaryOp::Add));
    }

    #[test]
    fn gpu_node_op_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<GpuNodeOp>(), 32);
    }
}
