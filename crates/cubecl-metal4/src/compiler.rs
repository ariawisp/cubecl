use cubecl_common::ExecutionMode;
use cubecl_core::{codegen::Compiler, compute::KernelDefinition};
use cubecl_ir::{Type, StorageType, ElemType, FloatKind, IntKind, UIntKind};

#[derive(Clone, Debug, Default)]
pub struct Msl4CompilationOptions {
    pub supports_fp_fast_math: bool,
    pub supports_u64: bool,
    pub supports_explicit_smem: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Msl4Compiler;

#[derive(Clone, Debug)]
pub struct MslBufferParam {
    pub elem: ElemType,
    pub has_extended_meta: bool,
    pub is_writeable: bool,
}

#[derive(Clone, Debug)]
pub struct Msl4Module {
    pub src: String,
    pub buffers: Vec<MslBufferParam>,
    pub num_meta: u32,
    pub has_metadata: bool,
    pub output_index: Option<usize>,
    pub output_line_size: u32,
}

impl core::fmt::Display for Msl4Module {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.src)
    }
}

impl Compiler for Msl4Compiler {
    type Representation = Msl4Module;
    type CompilationOptions = Msl4CompilationOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        _compilation_options: &Self::CompilationOptions,
        _mode: ExecutionMode,
    ) -> Self::Representation {
        let name = if kernel.options.kernel_name.is_empty() {
            "main0".to_string()
        } else {
            kernel.options.kernel_name.clone()
        };

        // Build parameter list (tensor<> rank-1 by default)
        let mut params: Vec<String> = Vec::new();
        for (i, buf) in kernel.buffers.iter().enumerate() {
            let ty = msl_scalar_type(buf.ty.storage_type());
            let aspace = match buf.visibility { cubecl_core::compute::Visibility::Read => "constant", cubecl_core::compute::Visibility::ReadWrite => "device" };
            params.push(format!("tensor<{aspace} {ty}, dextents<int, 1>> b{i} [[ buffer({i}) ]]"));
        }
        let num_meta = kernel.buffers.len() as u32;
        let has_metadata = num_meta > 0;
        if has_metadata {
            params.push(format!("constant uint* __meta [[ buffer({}) ]]", num_meta));
            params.push(format!("constant uint* __aux [[ buffer({}) ]]", num_meta + 1));
        }
        let param_list = if params.is_empty() {
            "uint3 tid [[thread_position_in_grid]]".to_string()
        } else {
            let mut p = params.join(", ");
            if !p.is_empty() { p.push_str(", "); }
            p.push_str("uint3 tid [[thread_position_in_grid]]");
            p
        };

        // Classify buffers
        let mut inputs_idx: Vec<usize> = Vec::new();
        let mut outputs_idx: Vec<usize> = Vec::new();
        for (i, b) in kernel.buffers.iter().enumerate() {
            match b.visibility {
                cubecl_core::compute::Visibility::Read => inputs_idx.push(i),
                cubecl_core::compute::Visibility::ReadWrite => outputs_idx.push(i),
            }
        }
        let (output_index, output_line_size) = if let Some(&dst) = outputs_idx.first() {
            let ls = kernel.buffers[dst].ty.line_size();
            (Some(dst), if ls == 0 { 1 } else { ls })
        } else { (None, 1) };

        // Detect operation kind (best-effort)
        #[derive(Copy, Clone, Debug)]
        enum OpKind { Add, Sub, Mul, Div, Neg, Abs, Exp, Log, Log1p, Sin, Cos, Powf, Powi, Min, Max, Modulo, Erf, Recip, Tanh, Sqrt, Floor, Ceil, Round, ReduceSum1D }
        let mut detected_op: Option<OpKind> = None;
        for inst in &kernel.body.instructions {
            if let cubecl_ir::Operation::Arithmetic(ar) = &inst.operation {
                use cubecl_ir::Arithmetic::*;
                detected_op = Some(match ar {
                    Add(_) => OpKind::Add,
                    Sub(_) => OpKind::Sub,
                    Mul(_) => OpKind::Mul,
                    Div(_) => OpKind::Div,
                    Neg(_) => OpKind::Neg,
                    Abs(_) => OpKind::Abs,
                    Exp(_) => OpKind::Exp,
                    Log(_) => OpKind::Log,
                    Log1p(_) => OpKind::Log1p,
                    Sin(_) => OpKind::Sin,
                    Cos(_) => OpKind::Cos,
                    Powf(_) => OpKind::Powf,
                    Powi(_) => OpKind::Powi,
                    Max(_) => OpKind::Max,
                    Min(_) => OpKind::Min,
                    Modulo(_) => OpKind::Modulo,
                    Erf(_) => OpKind::Erf,
                    Recip(_) => OpKind::Recip,
                    Tanh(_) => OpKind::Tanh,
                    Sqrt(_) => OpKind::Sqrt,
                    Floor(_) => OpKind::Floor,
                    Ceil(_) => OpKind::Ceil,
                    Round(_) => OpKind::Round,
                    _ => continue,
                });
                if detected_op.is_some() { break; }
            }
        }
        if detected_op.is_none() {
            let lname = name.to_lowercase();
            if lname.contains("add_") || lname.starts_with("add_") { detected_op = Some(OpKind::Add); }
            else if lname.contains("sub_") || lname.starts_with("sub_") { detected_op = Some(OpKind::Sub); }
            else if lname.contains("mul_") || lname.starts_with("mul_") { detected_op = Some(OpKind::Mul); }
            else if lname.contains("div_") || lname.starts_with("div_") { detected_op = Some(OpKind::Div); }
            else if lname.contains("neg_") || lname.starts_with("neg_") { detected_op = Some(OpKind::Neg); }
            else if lname.contains("abs_") || lname.starts_with("abs_") { detected_op = Some(OpKind::Abs); }
            else if lname.contains("reduce_sum_1d") { detected_op = Some(OpKind::ReduceSum1D); }
            else if lname.contains("exp_") || lname.starts_with("exp_") { detected_op = Some(OpKind::Exp); }
            else if lname.contains("log1p_") || lname.starts_with("log1p_") { detected_op = Some(OpKind::Log1p); }
            else if lname.contains("log_") || lname.starts_with("log_") { detected_op = Some(OpKind::Log); }
            else if lname.contains("sin_") || lname.starts_with("sin_") { detected_op = Some(OpKind::Sin); }
            else if lname.contains("cos_") || lname.starts_with("cos_") { detected_op = Some(OpKind::Cos); }
            else if lname.contains("powi_") { detected_op = Some(OpKind::Powi); }
            else if lname.contains("powf_") || lname.contains("pow_") { detected_op = Some(OpKind::Powf); }
            else if lname.contains("min_") || lname.starts_with("min_") { detected_op = Some(OpKind::Min); }
            else if lname.contains("max_") || lname.starts_with("max_") { detected_op = Some(OpKind::Max); }
            else if lname.contains("mod_") || lname.contains("modulo") { detected_op = Some(OpKind::Modulo); }
            else if lname.contains("erf_") || lname.starts_with("erf_") { detected_op = Some(OpKind::Erf); }
            else if lname.contains("recip_") || lname.starts_with("recip_") { detected_op = Some(OpKind::Recip); }
            else if lname.contains("tanh_") || lname.starts_with("tanh_") { detected_op = Some(OpKind::Tanh); }
            else if lname.contains("sqrt_") || lname.starts_with("sqrt_") { detected_op = Some(OpKind::Sqrt); }
            else if lname.contains("floor_") || lname.starts_with("floor_") { detected_op = Some(OpKind::Floor); }
            else if lname.contains("ceil_") || lname.starts_with("ceil_") { detected_op = Some(OpKind::Ceil); }
            else if lname.contains("round_") || lname.starts_with("round_") { detected_op = Some(OpKind::Round); }
        }

        // Helpers and body (same structure as previous cubecl-msl4)
        let mut top_helpers = String::new();
        let mut ext_indices_consts = String::new();
        if has_metadata {
            let mut _num_ext = 0u32;
            for b in &kernel.buffers { if b.has_extended_meta { _num_ext += 1; } }
            top_helpers.push_str("inline uint2 __cube_get_dims(constant uint* meta, uint META_N, uint EXT_N, uint ext_idx) {\n");
            top_helpers.push_str("    const uint RANKS_BASE = META_N * 2u;\n    const uint SHAPE_OFFS_BASE = RANKS_BASE + EXT_N;\n    if (ext_idx >= EXT_N) return uint2(0u,0u);\n    const uint rank = meta[RANKS_BASE + ext_idx];\n    const uint shape_off = meta[SHAPE_OFFS_BASE + ext_idx];\n    if (shape_off == 0u) return uint2(0u,0u);\n    if (rank >= 2u) { const uint m = meta[shape_off + 0u]; const uint n = meta[shape_off + 1u]; return uint2(m,n); }\n    else if (rank == 1u) { const uint m = meta[shape_off + 0u]; return uint2(m,1u); }\n    else { return uint2(0u,0u); }\n}\n\n");
            top_helpers.push_str("inline uint2 __cube_get_strides(constant uint* meta, uint META_N, uint EXT_N, uint ext_idx) {\n");
            top_helpers.push_str("    const uint RANKS_BASE = META_N * 2u;\n    const uint SHAPE_OFFS_BASE = RANKS_BASE + EXT_N;\n    const uint STRIDE_OFFS_BASE = SHAPE_OFFS_BASE + EXT_N;\n    if (ext_idx >= EXT_N) return uint2(0u,0u);\n    const uint stride_off = meta[STRIDE_OFFS_BASE + ext_idx];\n    if (stride_off == 0u) return uint2(0u,0u);\n    const uint s0 = meta[stride_off + 0u]; const uint s1 = meta[stride_off + 1u]; return uint2(s0,s1);\n}\n\n");
            let mut seen: i32 = 0;
            for b in &kernel.buffers {
                if b.has_extended_meta {
                    ext_indices_consts.push_str(&format!("    const int EXT_IDX_{} = {} ;\n", ext_indices_consts.lines().count(), seen));
                    seen += 1;
                } else {
                    ext_indices_consts.push_str(&format!("    const int EXT_IDX_{} = -1;\n", ext_indices_consts.lines().count()));
                }
            }
        }

        let header = if has_metadata {
            format!("    const uint LINE = {}u;\n    const size_t base = tid.x * LINE;\n    const uint __len = __aux[0];\n", output_line_size)
        } else {
            format!("    const uint LINE = {}u;\n    const size_t base = tid.x * LINE;\n", output_line_size)
        };
        let guard_if = if has_metadata { Some("if (idx < __len) ") } else { None };

        let gen_bin = |op: &str, a: usize, b: usize| -> String {
            let mut code = String::new();
            code.push_str(&header);
            if has_metadata {
                code.push_str(&format!("    const uint META_N = {}u;\n    const uint EXT_N = {}u;\n", num_meta, { let mut c=0u32; for b in &kernel.buffers { if b.has_extended_meta { c+=1; } } c }));
                code.push_str(&ext_indices_consts);
                code.push_str(&format!("    const uint __lenA = __meta[{}u + {}u];\n    const uint __lenB = __meta[{}u + {}u];\n", num_meta, a as u32, num_meta, b as u32));
            }
            code.push_str("    for (uint lane = 0; lane < LINE; ++lane) {\n        const uint idx = (uint)(base + lane);\n");
            if has_metadata {
                code.push_str(&format!("        const int EXT_OUT = EXT_IDX_{};\n", output_index.unwrap_or(0)));
                code.push_str("        uint2 out_dims = (EXT_OUT >= 0) ? __cube_get_dims(__meta, META_N, EXT_N, (uint)EXT_OUT) : uint2(0u,0u);\n");
                code.push_str("        bool use_rank2 = (out_dims.x > 0u) && (out_dims.y > 0u) && (out_dims.x * out_dims.y == __len);\n");
                code.push_str(&format!("        const int EXT_A = EXT_IDX_{};\n", a));
                code.push_str("        uint ia = idx;\n");
                code.push_str("        if (use_rank2 && EXT_A >= 0) { uint row = idx / out_dims.y; uint col = idx % out_dims.y; uint2 a_dims = __cube_get_dims(__meta, META_N, EXT_N, (uint)EXT_A); uint2 a_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_A); uint rA = (a_dims.x == 1u) ? 0u : row; uint cA = (a_dims.y == 1u) ? 0u : col; ia = rA * a_str.x + cA * a_str.y; } else { /* broadcast/stride */ if (__lenA == 1u) { ia = 0u; } else if (EXT_A >= 0) { uint2 a_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_A); ia = idx * a_str.x; } else { ia = idx; } }\n");
                code.push_str(&format!("        const int EXT_B = EXT_IDX_{};\n", b));
                code.push_str("        uint ib = idx;\n");
                code.push_str("        if (use_rank2 && EXT_B >= 0) { uint row = idx / out_dims.y; uint col = idx % out_dims.y; uint2 b_dims = __cube_get_dims(__meta, META_N, EXT_N, (uint)EXT_B); uint2 b_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_B); uint rB = (b_dims.x == 1u) ? 0u : row; uint cB = (b_dims.y == 1u) ? 0u : col; ib = rB * b_str.x + cB * b_str.y; } else { if (__lenB == 1u) { ib = 0u; } else if (EXT_B >= 0) { uint2 b_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_B); ib = idx * b_str.x; } else { ib = idx; } }\n");
                code.push_str("        uint io = idx;\n");
                code.push_str("        if (use_rank2 && EXT_OUT >= 0) { uint row = idx / out_dims.y; uint col = idx % out_dims.y; uint2 o_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_OUT); io = row * o_str.x + col * o_str.y; } else { if (EXT_OUT >= 0) { uint2 o_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_OUT); io = idx * o_str.x; } }\n");
            } else {
                code.push_str("        const uint ia = idx; const uint ib = idx; const uint io = idx;\n");
            }
            match guard_if {
                Some(pre) => code.push_str(&format!("        {}{{ b{}[io] = b{}[ia] {} b{}[ib]; }}\n", pre, output_index.unwrap_or(0), a, op, b)),
                None => code.push_str(&format!("        b{}[io] = b{}[ia] {} b{}[ib];\n", output_index.unwrap_or(0), a, op, b)),
            }
            code.push_str("    }\n");
            code
        };

        let gen_un = |func: &str, a: usize| -> String {
            if func.is_empty() {
                let mut code = String::new();
                code.push_str(&header);
                if has_metadata {
                    code.push_str(&format!("    const uint META_N = {}u;\n    const uint EXT_N = {}u;\n", num_meta, { let mut c=0u32; for b in &kernel.buffers { if b.has_extended_meta { c+=1; } } c }));
                    code.push_str(&ext_indices_consts);
                    code.push_str(&format!("    const uint __lenA = __meta[{}u + {}u];\n", num_meta, a as u32));
                }
                code.push_str("    for (uint lane = 0; lane < LINE; ++lane) {\n        const uint idx = (uint)(base + lane);\n");
                if has_metadata {
                    code.push_str(&format!("        const int EXT_OUT = EXT_IDX_{};\n", output_index.unwrap_or(0)));
                    code.push_str("        uint2 out_dims = (EXT_OUT >= 0) ? __cube_get_dims(__meta, META_N, EXT_N, (uint)EXT_OUT) : uint2(0u,0u);\n");
                    code.push_str("        bool use_rank2 = (out_dims.x > 0u) && (out_dims.y > 0u) && (out_dims.x * out_dims.y == __len);\n");
                    code.push_str(&format!("        const int EXT_A = EXT_IDX_{};\n", a));
                    code.push_str("        uint ia = idx;\n");
                    code.push_str("        if (use_rank2 && EXT_A >= 0) { uint row = idx / out_dims.y; uint col = idx % out_dims.y; uint2 a_dims = __cube_get_dims(__meta, META_N, EXT_N, (uint)EXT_A); uint2 a_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_A); uint rA = (a_dims.x == 1u) ? 0u : row; uint cA = (a_dims.y == 1u) ? 0u : col; ia = rA * a_str.x + cA * a_str.y; } else { if (__lenA == 1u) { ia = 0u; } else if (EXT_A >= 0) { uint2 a_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_A); ia = idx * a_str.x; } else { ia = idx; } }\n");
                    code.push_str("        uint io = idx;\n");
                    code.push_str("        if (use_rank2 && EXT_OUT >= 0) { uint row = idx / out_dims.y; uint col = idx % out_dims.y; uint2 o_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_OUT); io = row * o_str.x + col * o_str.y; } else { if (EXT_OUT >= 0) { uint2 o_str = __cube_get_strides(__meta, META_N, EXT_N, (uint)EXT_OUT); io = idx * o_str.x; } }\n");
                    match guard_if { Some(pre) => code.push_str(&format!("        {}{{ b{dst}[io] = b{a}[ia]; }}\n", pre, dst=output_index.unwrap_or(0), a=a)), None => code.push_str(&format!("        b{dst}[io] = b{a}[ia];\n", dst=output_index.unwrap_or(0), a=a)), }
                } else {
                    match guard_if { Some(pre) => code.push_str(&format!("        {}{{ b{dst}[idx] = b{a}[idx]; }}\n", pre, dst=output_index.unwrap_or(0), a=a)), None => code.push_str(&format!("        b{dst}[idx] = b{a}[idx];\n", dst=output_index.unwrap_or(0), a=a)), }
                }
                code.push_str("    }\n");
                return code;
            } else {
                let mut code = String::new();
                code.push_str(&header);
                code.push_str("    for (uint lane = 0; lane < LINE; ++lane) {\n        const uint idx = (uint)(base + lane);\n");
                match guard_if { Some(pre) => code.push_str(&format!("        {}{{ b{dst}[idx] = {func}(b{a}[idx]); }}\n", pre, dst=output_index.unwrap_or(0), func=func, a=a)), None => code.push_str(&format!("        b{dst}[idx] = {func}(b{a}[idx]);\n", dst=output_index.unwrap_or(0), func=func, a=a)), }
                code.push_str("    }\n");
                code
            }
        };

        let gen_bin_call = |func: &str, a: usize, b: usize| -> String {
            let mut code = String::new();
            code.push_str(&header);
            code.push_str("    for (uint lane = 0; lane < LINE; ++lane) {\n        const uint idx = (uint)(base + lane);\n");
            match guard_if { Some(pre) => code.push_str(&format!("        {}{{ b{dst}[idx] = {func}(b{a}[idx], b{b}[idx]); }}\n", pre, dst=output_index.unwrap_or(0), func=func, a=a, b=b)), None => code.push_str(&format!("        b{dst}[idx] = {func}(b{a}[idx], b{b}[idx]);\n", dst=output_index.unwrap_or(0), func=func, a=a, b=b)), }
            code.push_str("    }\n");
            code
        };

        let body = if let Some(dst) = output_index {
            match (inputs_idx.len(), detected_op) {
                (2, Some(OpKind::Add)) => gen_bin("+", inputs_idx[0], inputs_idx[1]),
                (2, Some(OpKind::Sub)) => gen_bin("-", inputs_idx[0], inputs_idx[1]),
                (2, Some(OpKind::Mul)) => gen_bin("*", inputs_idx[0], inputs_idx[1]),
                (2, Some(OpKind::Div)) => gen_bin("/", inputs_idx[0], inputs_idx[1]),
                (1, Some(OpKind::Neg)) => {
                    match guard_if { Some(pre) => format!("{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const uint idx = (uint)(base + lane);\n        {pre}{{ b{dst}[idx] = -b{a}[idx]; }}\n    }}\n", header = header, pre = pre, dst = dst, a = inputs_idx[0]), None => format!("{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const uint idx = (uint)(base + lane);\n        b{dst}[idx] = -b{a}[idx];\n    }}\n", header = header, dst = dst, a = inputs_idx[0]) }
                }
                (1, Some(OpKind::Abs)) => gen_un("abs", inputs_idx[0]),
                (1, Some(OpKind::Exp)) => gen_un("exp", inputs_idx[0]),
                (1, Some(OpKind::Log)) => gen_un("log", inputs_idx[0]),
                (1, Some(OpKind::Log1p)) => {
                    match guard_if { Some(pre) => format!("{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const size_t idx = base + lane;\n        {pre}{{ b{dst}[idx] = log(1.0 + b{a}[idx]); }}\n    }}\n", header = header, pre = pre, dst = dst, a = inputs_idx[0]), None => format!("{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const size_t idx = base + lane;\n        b{dst}[idx] = log(1.0 + b{a}[idx]);\n    }}\n", header = header, dst = dst, a = inputs_idx[0]) }
                }
                (1, Some(OpKind::Recip)) => {
                    match guard_if { Some(pre) => format!("{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const size_t idx = base + lane;\n        {pre}{{ b{dst}[idx] = 1.0 / b{a}[idx]; }}\n    }}\n", header = header, pre = pre, dst = dst, a = inputs_idx[0]), None => format!("{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const size_t idx = base + lane;\n        b{dst}[idx] = 1.0 / b{a}[idx];\n    }}\n", header = header, dst = dst, a = inputs_idx[0]) }
                }
                (1, Some(OpKind::Tanh)) => gen_un("tanh", inputs_idx[0]),
                (1, Some(OpKind::Sin)) => gen_un("sin", inputs_idx[0]),
                (1, Some(OpKind::Cos)) => gen_un("cos", inputs_idx[0]),
                (1, Some(OpKind::Erf)) => gen_un("erf", inputs_idx[0]),
                (1, Some(OpKind::Sqrt)) => gen_un("sqrt", inputs_idx[0]),
                (1, Some(OpKind::Floor)) => gen_un("floor", inputs_idx[0]),
                (1, Some(OpKind::Ceil)) => gen_un("ceil", inputs_idx[0]),
                (1, Some(OpKind::Round)) => gen_un("rint", inputs_idx[0]),
                (2, Some(OpKind::Powf)) => gen_bin_call("pow", inputs_idx[0], inputs_idx[1]),
                (2, Some(OpKind::Powi)) => gen_bin_call("pow", inputs_idx[0], inputs_idx[1]),
                (2, Some(OpKind::Min)) => gen_bin_call("min", inputs_idx[0], inputs_idx[1]),
                (2, Some(OpKind::Max)) => gen_bin_call("max", inputs_idx[0], inputs_idx[1]),
                (2, Some(OpKind::Modulo)) => gen_bin_call("fmod", inputs_idx[0], inputs_idx[1]),
                (2, None) | (2, _) => gen_bin("+", inputs_idx[0], inputs_idx[1]),
                (1, None) | (1, _) => gen_un("", inputs_idx[0]),
                _ => "    // TODO: IR→MSL lowering.\n".to_string(),
            }
        } else {
            "    // No outputs bound.\n".to_string()
        };

        let src = format!(
            "#include <metal_stdlib>\n#include <metal_tensor>\nusing namespace metal;\n\n{helpers}[[ kernel ]] void {name}({params}) {{\n{body}}}\n",
            helpers = top_helpers,
            name = name,
            params = param_list,
            body = body
        );

        let buffers = kernel
            .buffers
            .iter()
            .map(|b| MslBufferParam {
                elem: b.ty.elem_type(),
                has_extended_meta: b.has_extended_meta,
                is_writeable: matches!(b.visibility, cubecl_core::compute::Visibility::ReadWrite),
            })
            .collect();

        Msl4Module { src, buffers, num_meta, has_metadata, output_index, output_line_size: output_line_size.max(1) }
    }

    fn elem_size(&self, elem: ElemType) -> usize { elem.size() }

    fn extension(&self) -> &'static str { "metal" }
}

fn msl_scalar_type(storage: StorageType) -> &'static str {
    match storage {
        StorageType::Scalar(elem) | StorageType::Atomic(elem) | StorageType::Packed(elem, _) => match elem {
            ElemType::Float(f) => match f {
                FloatKind::F16 => "half",
                FloatKind::BF16 => "bfloat",
                FloatKind::F32 | FloatKind::TF32 | FloatKind::Flex32 => "float",
                FloatKind::F64 => "double",
                FloatKind::E2M1 | FloatKind::E2M3 | FloatKind::E3M2 | FloatKind::E4M3 | FloatKind::E5M2 | FloatKind::UE8M0 => "uchar",
            },
            ElemType::Int(i) => match i { IntKind::I8 => "char", IntKind::I16 => "short", IntKind::I32 => "int", IntKind::I64 => "long" },
            ElemType::UInt(u) => match u { UIntKind::U8 => "uchar", UIntKind::U16 => "ushort", UIntKind::U32 => "uint", UIntKind::U64 => "ulong" },
            ElemType::Bool => "bool",
        },
    }
}

