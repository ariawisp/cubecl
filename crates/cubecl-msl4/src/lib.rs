use cubecl_common::ExecutionMode;
use cubecl_core::{codegen::Compiler, compute::KernelDefinition};
use cubecl_ir::ElemType;

#[derive(Clone, Debug, Default)]
pub struct Msl4CompilationOptions {
    pub supports_fp_fast_math: bool,
    pub supports_u64: bool,
    pub supports_explicit_smem: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Msl4Compiler;

impl Compiler for Msl4Compiler {
    type Representation = String;
    type CompilationOptions = Msl4CompilationOptions;

    fn compile(
        &mut self,
        _kernel: KernelDefinition,
        _compilation_options: &Self::CompilationOptions,
        _mode: ExecutionMode,
    ) -> Self::Representation {
        // NOTE: This is a scaffold. A full IR->MSL 4.0 compiler will be implemented.
        // For now, emit a minimal valid kernel entry as placeholder.
        // Metal 4 requires language version selection at compile time via MTL4Compiler.
        "kernel void main0() {}".to_string()
    }

    fn elem_size(&self, elem: ElemType) -> usize { elem.size() }

    fn extension(&self) -> &'static str {
        "metal"
    }
}
