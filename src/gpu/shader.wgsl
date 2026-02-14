// bevy_autodiff GPU interpreter kernel
//
// Evaluates a compiled computation graph at many input points in parallel.
// Each GPU thread processes one sample, executing the same node sequence.
// Since all threads follow the same control flow, there is no warp divergence.
//
// Memory layout is Structure-of-Arrays (SoA):
//   values[node_idx * num_samples + sample_id]
// This gives coalesced memory access — adjacent threads read adjacent addresses.

struct Params {
    num_nodes: u32,
    num_samples: u32,
    num_inputs: u32,
    num_outputs: u32,
}

struct Node {
    op_type: u32,    // 0=Input, 1=Constant, 2=Unary, 3=Binary
    op_code: u32,    // operation discriminant or input index
    arg1: u32,       // src (unary), lhs (binary)
    arg2: u32,       // rhs (binary)
    const_val: f32,  // constant value
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> inputs: array<f32>;
@group(0) @binding(3) var<storage, read_write> values: array<f32>;
@group(0) @binding(4) var<storage, read> output_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputs: array<f32>;

fn eval_unary(op_code: u32, x: f32) -> f32 {
    switch op_code {
        case 0u  { return -x; }           // Neg
        case 1u  { return sin(x); }       // Sin
        case 2u  { return cos(x); }       // Cos
        case 3u  { return tan(x); }       // Tan
        case 4u  { return exp(x); }       // Exp
        case 5u  { return log(x); }       // Ln (WGSL log = natural log)
        case 6u  { return sqrt(x); }      // Sqrt
        case 7u  { return sinh(x); }      // Sinh
        case 8u  { return cosh(x); }      // Cosh
        case 9u  { return tanh(x); }      // Tanh
        case 10u { return asin(x); }      // Asin
        case 11u { return acos(x); }      // Acos
        case 12u { return atan(x); }      // Atan
        case 13u { return asinh(x); }     // Asinh
        case 14u { return acosh(x); }     // Acosh
        case 15u { return atanh(x); }     // Atanh
        default  { return 0.0; }
    }
}

fn eval_binary(op_code: u32, x: f32, y: f32) -> f32 {
    switch op_code {
        case 0u  { return x + y; }        // Add
        case 1u  { return x - y; }        // Sub
        case 2u  { return x * y; }        // Mul
        case 3u  { return x / y; }        // Div
        case 4u  { return pow(x, y); }    // Pow
        default  { return 0.0; }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sid = gid.x;
    if sid >= params.num_samples {
        return;
    }

    // Forward pass: interpret node array sequentially
    for (var i: u32 = 0u; i < params.num_nodes; i = i + 1u) {
        let node = nodes[i];
        var val: f32 = 0.0;

        switch node.op_type {
            case 0u {
                // Input: read from inputs buffer (SoA)
                val = inputs[node.op_code * params.num_samples + sid];
            }
            case 1u {
                // Constant
                val = node.const_val;
            }
            case 2u {
                // Unary
                let src = values[node.arg1 * params.num_samples + sid];
                val = eval_unary(node.op_code, src);
            }
            case 3u {
                // Binary
                let lhs = values[node.arg1 * params.num_samples + sid];
                let rhs = values[node.arg2 * params.num_samples + sid];
                val = eval_binary(node.op_code, lhs, rhs);
            }
            default {}
        }

        values[i * params.num_samples + sid] = val;
    }

    // Gather requested outputs into compact output buffer
    for (var o: u32 = 0u; o < params.num_outputs; o = o + 1u) {
        let node_idx = output_indices[o];
        outputs[o * params.num_samples + sid] = values[node_idx * params.num_samples + sid];
    }
}
