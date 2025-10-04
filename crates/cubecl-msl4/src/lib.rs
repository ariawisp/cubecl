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
        // Minimal MSL 4.0 kernel scaffold
        // - Emits a kernel function with one parameter per bound buffer.
        // - Uses raw bytes (uchar*) for maximum compatibility until typed lowering lands.
        // - No-op body; serves to validate pipeline, binding order, and dispatch sizing.

        let name = if kernel.options.kernel_name.is_empty() {
            "main0".to_string()
        } else {
            kernel.options.kernel_name.clone()
        };

        // For elementwise kernels (non-MPP), use typed tensor params; MPP uses built-in tensor kernels
        let mut params: Vec<String> = Vec::new();
        for (i, buf) in kernel.buffers.iter().enumerate() {
            let ty = msl_scalar_type(buf.ty.storage_type());
            let aspace = match buf.visibility { cubecl_core::compute::Visibility::Read => "constant", cubecl_core::compute::Visibility::ReadWrite => "device" };
            // Proper MTLTensor path: typed tensor params for rank-1
            params.push(format!("tensor<{aspace} {ty}, dextents<int, 1>> b{i} [[ buffer({i}) ]]"));
        }
        // Metadata binding: when present, expose as a raw constant u32 array
        // Layout follows cubecl-core/src/codegen/metadata.rs
        let num_meta = kernel.buffers.len() as u32;
        let has_metadata = num_meta > 0;
        if has_metadata {
            // Metadata is bound right after the last buffer index
            params.push(format!("constant uint* __meta [[ buffer({}) ]]", num_meta));
        }

        let param_list = if params.is_empty() {
            // MSL requires at least the thread index param
            "uint3 tid [[thread_position_in_grid]]".to_string()
        } else {
            let mut p = params.join(", ");
            if !p.is_empty() {
                p.push_str(", ");
            }
            p.push_str("uint3 tid [[thread_position_in_grid]]");
            p
        };

        // Classify buffers by visibility: inputs are Read, outputs are ReadWrite
        let mut inputs_idx: Vec<usize> = Vec::new();
        let mut outputs_idx: Vec<usize> = Vec::new();
        for (i, b) in kernel.buffers.iter().enumerate() {
            match b.visibility {
                cubecl_core::compute::Visibility::Read => inputs_idx.push(i),
                cubecl_core::compute::Visibility::ReadWrite => outputs_idx.push(i),
            }
        }
        // Determine output vectorization factor (line size) if any
        let (output_index, output_line_size) = if let Some(&dst) = outputs_idx.first() {
            let ls = kernel.buffers[dst].ty.line_size();
            (Some(dst), if ls == 0 { 1 } else { ls })
        } else { (None, 1) };

        // Emit basic elementwise on buffers (use operator[] for tensors)
        // Vectorized per-lane loop guarded by logical length from metadata
        let body = if let Some(dst) = output_index {
            let header = if has_metadata {
                format!(
                    "    const uint LINE = {}u;\n    const size_t base = tid.x * LINE;\n    const uint __len = __meta[{}u + {}u];\n",
                    output_line_size,
                    num_meta, // LENGTH block starts at offset num_meta
                    dst as u32,
                )
            } else {
                format!(
                    "    const uint LINE = {}u;\n    const size_t base = tid.x * LINE;\n",
                    output_line_size
                )
            };

            if inputs_idx.len() >= 2 {
                let a = inputs_idx[0];
                let b = inputs_idx[1];
                if has_metadata {
                    format!(
                        "{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const size_t idx = base + lane;\n        if (idx < __len) {{ b{dst}[idx] = b{a}[idx] + b{b}[idx]; }}\n    }}\n",
                        header = header,
                        dst = dst,
                        a = a,
                        b = b,
                    )
                } else {
                    format!(
                        "{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const size_t idx = base + lane;\n        b{dst}[idx] = b{a}[idx] + b{b}[idx];\n    }}\n",
                        header = header,
                        dst = dst,
                        a = a,
                        b = b,
                    )
                }
            } else if inputs_idx.len() == 1 {
                let src = inputs_idx[0];
                if has_metadata {
                    format!(
                        "{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const size_t idx = base + lane;\n        if (idx < __len) {{ b{dst}[idx] = b{src}[idx]; }}\n    }}\n",
                        header = header,
                        dst = dst,
                        src = src,
                    )
                } else {
                    format!(
                        "{header}    for (uint lane = 0; lane < LINE; ++lane) {{\n        const size_t idx = base + lane;\n        b{dst}[idx] = b{src}[idx];\n    }}\n",
                        header = header,
                        dst = dst,
                        src = src,
                    )
                }
            } else {
                "    // TODO: IR→MSL lowering.\n".to_string()
            }
        } else {
            // No outputs: nothing to do
            "    // No outputs bound.\n".to_string()
        };

        let src = format!(
            "#include <metal_stdlib>\n#include <metal_tensor>\nusing namespace metal;\n\n[[ kernel ]] void {name}({params}) {{\n{body}}}\n",
            name = name,
            params = param_list,
            body = body
        );

        // Collect buffer params (element types and meta flags)
        let buffers = kernel
            .buffers
            .iter()
            .map(|b| MslBufferParam {
                elem: b.ty.elem_type(),
                has_extended_meta: b.has_extended_meta,
                is_writeable: matches!(b.visibility, cubecl_core::compute::Visibility::ReadWrite),
            })
            .collect();

        Msl4Module {
            src,
            buffers,
            num_meta,
            has_metadata,
            output_index,
            output_line_size: output_line_size.max(1),
        }
    }

    fn elem_size(&self, elem: ElemType) -> usize { elem.size() }

    fn extension(&self) -> &'static str {
        "metal"
    }
}

fn msl_scalar_type(storage: StorageType) -> &'static str {
    match storage {
        StorageType::Scalar(elem) | StorageType::Atomic(elem) | StorageType::Packed(elem, _) => match elem {
            ElemType::Float(f) => match f {
                FloatKind::F16 => "half",
                FloatKind::BF16 => "bfloat",
                FloatKind::F32 | FloatKind::TF32 | FloatKind::Flex32 => "float",
                FloatKind::F64 => "double",
                // Map minifloats to uchar storage until dedicated lowering lands
                FloatKind::E2M1 | FloatKind::E2M3 | FloatKind::E3M2 | FloatKind::E4M3 | FloatKind::E5M2 | FloatKind::UE8M0 => "uchar",
            },
            ElemType::Int(i) => match i {
                IntKind::I8 => "char",
                IntKind::I16 => "short",
                IntKind::I32 => "int",
                IntKind::I64 => "long",
            },
            ElemType::UInt(u) => match u {
                UIntKind::U8 => "uchar",
                UIntKind::U16 => "ushort",
                UIntKind::U32 => "uint",
                UIntKind::U64 => "ulong",
            },
            ElemType::Bool => "bool",
        },
    }
}

fn msl_vector_type(storage: StorageType, n: u32) -> String {
    let base = msl_scalar_type(storage);
    match n {
        2 | 3 | 4 => format!("{base}{n}"),
        _ => base.to_string(),
    }
}

fn msl_ptr_type(ty: Type) -> String {
    match ty {
        Type::Scalar(st) => msl_scalar_type(st).to_string(),
        Type::Line(st, n) => msl_vector_type(st, n),
        Type::Semantic(_) => "uchar".to_string(), // should not occur for buffer bindings
    }
}
