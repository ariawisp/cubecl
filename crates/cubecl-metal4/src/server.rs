extern crate alloc;
use alloc::sync::Arc;
use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::{ProfileDuration, TimingMethod};
use cubecl_common::stream_id::StreamId;
use cubecl_core::server::{Allocation, AllocationDescriptor, IoError};
use cubecl_core::compute::CubeTask;
use cubecl_core::{
    MemoryConfiguration,
    prelude::*,
    server::{Binding, Bindings, CopyDescriptor, ProfileError, ProfilingToken},
};
use cubecl_msl4::Msl4Compiler;
use cubecl_runtime::config::GlobalConfig;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::MemoryDeviceProperties;
use cubecl_runtime::server::ComputeServer;
use cubecl_runtime::storage::{BindingResource, ComputeStorage};

use crate::storage::{Metal4Resource, Metal4Storage};
use hashbrown::HashMap;
use objc2::rc::Retained;
use objc2::ClassType;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandBuffer, MTL4CommandQueue,
    MTL4CommandEncoder, MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputeCommandEncoder, MTL4ComputePipelineDescriptor,
    MTL4FunctionDescriptor, MTL4LibraryFunctionDescriptor, MTLCommandEncoder, MTLCompileOptions,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLGPUAddress, MTLLibrary,
    MTLSize,
};
use core::ptr::NonNull;
use cubecl_runtime::id::KernelId;

/// Metal 4 compute server (scaffold).
#[derive(Debug)]
pub struct Metal4Server {
    pub(crate) logger: Arc<ServerLogger>,
    pub(crate) timing_method: TimingMethod,
    pub(crate) memory_properties: MemoryDeviceProperties,
    pub(crate) memory_config: MemoryConfiguration,
    pub(crate) storage: Metal4Storage,
    device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    pipelines: HashMap<KernelId, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

// Mark server Send/Sync: Metal device/queue are safe to share across threads for command creation.
unsafe impl Send for Metal4Server {}
unsafe impl Sync for Metal4Server {}

impl Metal4Server {
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        timing_method: TimingMethod,
        logger: Arc<ServerLogger>,
    ) -> Self {
        let _config = GlobalConfig::get();
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
        let queue = device
            .newMTL4CommandQueue()
            .expect("Failed to create MTL4CommandQueue");
        let alignment = memory_properties.alignment as usize;
        Self {
            logger,
            timing_method,
            memory_properties,
            memory_config,
            storage: Metal4Storage::new(device.clone(), alignment),
            device,
            queue,
            pipelines: HashMap::new(),
        }
    }
}

impl cubecl_core::server::ServerCommunication for Metal4Server {
    const SERVER_COMM_ENABLED: bool = false;
}

impl ComputeServer for Metal4Server {
    type Kernel = Box<dyn CubeTask<Msl4Compiler>>;
    type Storage = Metal4Storage;
    type Info = (&'static str,);

    fn logger(&self) -> Arc<ServerLogger> {
        self.logger.clone()
    }

    fn create(
        &mut self,
        _descriptors: Vec<AllocationDescriptor<'_>>,
        _stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        Err(IoError::Unknown(
            "Metal4Server::create not yet implemented".to_string(),
        ))
    }

    fn read<'a>(
        &mut self,
        _descriptors: Vec<CopyDescriptor<'a>>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        Box::pin(async { Err(IoError::UnsupportedStrides) })
    }

    fn write(
        &mut self,
        _descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        _stream_id: StreamId,
    ) -> Result<(), IoError> {
        // TODO: Implement staging/shared writes.
        Ok(())
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<()> {
        Box::pin(async { () })
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        _stream_id: StreamId,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource> {
        // Placeholder until memory pool is wired to Metal 4 resources
        BindingResource::new(
            binding,
            Metal4Resource { size: 0, offset: 0, storage_id: cubecl_runtime::storage::StorageId::new(), gpu_address: 0 },
        )
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        _bindings: Bindings,
        kind: ExecutionMode,
        _stream_id: StreamId,
    ) {
        // Compile to MSL and create pipeline if needed
        let mut compiler = Msl4Compiler::default();
        let compile = kernel.compile(&mut compiler, &Default::default(), kind);

        let mut kid = kernel.id();
        kid.mode(kind);

        let pso = if let Some(p) = self.pipelines.get(&kid) {
            p.clone()
        } else {
            // Compile source to library for function reference
            let opts = MTLCompileOptions::new();
            let src = NSString::from_str(&compile.source);
            let library: Retained<ProtocolObject<dyn MTLLibrary>> = self
                .device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .expect("Failed to compile MSL");

            // Try Metal 4 compiler path first
            let pso = (|| {
                let comp_desc = MTL4CompilerDescriptor::new();
                let compiler: Retained<ProtocolObject<dyn MTL4Compiler>> = self
                    .device
                    .newCompilerWithDescriptor_error(&comp_desc)
                    .ok()?;

                let lfd = MTL4LibraryFunctionDescriptor::new();
                let fname = NSString::from_str(&compile.entrypoint_name);
                lfd.setName(Some(&fname));
                lfd.setLibrary(Some(&library));

                let cpdesc = MTL4ComputePipelineDescriptor::new();
                // Upcast to base function descriptor
                let base: &MTL4FunctionDescriptor = lfd.as_super();
                cpdesc.setComputeFunctionDescriptor(Some(base));

                let pso = compiler
                    .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&cpdesc, None)
                    .ok()?;
                Some(pso)
            })()
            .unwrap_or_else(|| {
                // Fallback to older API if MTL4Compiler path is unavailable
                let fname = NSString::from_str(&compile.entrypoint_name);
                let function = library
                    .newFunctionWithName(&fname)
                    .expect("Missing entrypoint in library");
                self
                    .device
                    .newComputePipelineStateWithFunction_error(&function)
                    .expect("Failed to create compute pipeline state")
            });
            self.pipelines.insert(kid.clone(), pso.clone());
            pso
        };

        // Create MTL4 command buffer and encoder
        let Some(cb) = self.device.newCommandBuffer() else { return; };
        let Some(encoder) = cb.computeCommandEncoder() else { return; };
        encoder.setComputePipelineState(&pso);

        // Create MTL4 argument table sized to buffer bindings
        let at_desc = MTL4ArgumentTableDescriptor::new();
        at_desc.setMaxBufferBindCount(_bindings.buffers.len() as NSUInteger);
        if let Ok(arg_table) = self.device.newArgumentTableWithDescriptor_error(&at_desc) {
            // TODO: set GPU addresses with offsets once storage mapping is fully wired
            encoder.setArgumentTable(Some(&arg_table));
        }

        // TODO: Bind addresses per buffer in argument table; bind metadata/scalars as needed

        // Compute grid and threads
        let (gx, gy, gz) = match count {
            CubeCount::Static(x, y, z) => (x as usize, y as usize, z as usize),
            CubeCount::Dynamic(_) => (1, 1, 1),
        };
        let threads = MTLSize { width: compile.cube_dim.x as usize, height: compile.cube_dim.y as usize, depth: compile.cube_dim.z as usize };
        let grid = MTLSize { width: gx, height: gy, depth: gz };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
        encoder.endEncoding();
        // Commit via MTL4CommandQueue
        let mut one: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; 1] = [
            NonNull::new(Retained::as_ptr(&cb) as *mut _).expect("nonnull"),
        ];
        let arr = NonNull::new(one.as_mut_ptr()).expect("nonnull array ptr");
        unsafe { self.queue.commit_count(arr, 1) };
    }

    fn flush(&mut self, _stream_id: StreamId) {}

    fn memory_usage(&mut self, _stream_id: StreamId) -> cubecl_runtime::memory_management::MemoryUsage {
        // Placeholder; will report proper allocator stats.
        cubecl_runtime::memory_management::MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        }
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {}

    fn start_profile(&mut self, _stream_id: StreamId) -> ProfilingToken {
        ProfilingToken { id: 0 }
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        _token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        Err(ProfileError::NotRegistered)
    }

    fn allocation_mode(&mut self, _mode: cubecl_runtime::memory_management::MemoryAllocationMode, _stream_id: StreamId) {}
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
