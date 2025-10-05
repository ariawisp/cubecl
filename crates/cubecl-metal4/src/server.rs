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
use crate::compiler::Msl4Compiler;
use cubecl_runtime::config::GlobalConfig;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::MemoryDeviceProperties;
use cubecl_runtime::server::ComputeServer;
use cubecl_runtime::storage::{BindingResource, ComputeStorage};

use crate::storage::Metal4Storage;
use hashbrown::HashMap;
use objc2::rc::Retained;
use objc2::ClassType;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSInteger, NSUInteger};
use objc2::AnyThread;
use objc2_metal::MTLBuffer;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandQueue,
    MTL4CommandEncoder, MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputeCommandEncoder, MTL4ComputePipelineDescriptor,
    MTL4FunctionDescriptor, MTL4LibraryFunctionDescriptor, MTLCommandEncoder, MTLCompileOptions,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLGPUAddress, MTLLibrary,
    MTLSize,
};
// Legacy command queue API not used in MTL4 path
use objc2_metal::{MTLEvent, MTLSharedEvent};
use objc2_metal::{MTLResidencySet, MTLResidencySetDescriptor, MTLAllocation};
// Allocator used implicitly when beginning MTL4 command buffer
// MTLTensor API (Metal 4)
use objc2_metal::{MTLTensor, MTLTensorDescriptor, MTLTensorExtents, MTLTensorDataType, MTLTensorUsage};
use core::ptr::NonNull;
use cubecl_runtime::id::KernelId;
use cubecl_runtime::server::Handle;
use cubecl_runtime::memory_management::{MemoryManagement, MemoryDeviceProperties as MMProps};
use cubecl_ir::ElemType;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Msl4SpecKey {
    entry: alloc::string::String,
    line: u32,
    types: alloc::vec::Vec<cubecl_ir::ElemType>,
    fast_math: bool,
}
/// Metal 4 compute server (scaffold).
#[derive(Debug)]
pub struct Metal4Server {
    pub(crate) logger: Arc<ServerLogger>,
    pub(crate) timing_method: TimingMethod,
    pub(crate) memory_properties: MemoryDeviceProperties,
    pub(crate) memory_config: MemoryConfiguration,
    mem_manage: MemoryManagement<Metal4Storage>,
    device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    fence_value: u64,
    // Reuse a command buffer with a small allocator ring to reduce churn
    cb: Option<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>,
    allocators: Vec<Retained<ProtocolObject<dyn MTL4CommandAllocator>>>,
    frames_in_flight: usize,
    frame_cursor: usize,
    // Optional batching: defer commit and submit multiple command buffers as a group
    pending_cbs: Vec<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>,
    defer_commit: bool,
    inflight_staging: Vec<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    inflight_fences: Vec<u64>,
    inflight_tables: Vec<Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>>,
    inflight_tensors: Vec<Vec<Retained<ProtocolObject<dyn MTLTensor>>>>,
    // Pipeline cache
    pipelines: HashMap<KernelId, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    // Pointer fallback removed: typed tensors are required for MTL4 path.
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
        // Detect MTL4 support; fail fast with a clear message if unavailable
        let queue = match device.newMTL4CommandQueue() {
            Some(q) => q,
            None => {
                panic!(
                    "Metal 4 unsupported on this device/OS. Enable an alternate runtime (e.g., WGPU) or provide a Metal 3 path."
                );
            }
        };
        // Create a residency set and add it to the queue (keeps allocations resident)
        let res_desc = MTLResidencySetDescriptor::new();
        let residency = device
            .newResidencySetWithDescriptor_error(&res_desc)
            .expect("Failed to create MTLResidencySet");
        queue.addResidencySet(&residency);
        let event = device
            .newSharedEvent()
            .expect("Failed to create MTLSharedEvent");
        let alignment = memory_properties.alignment as usize;
        let mm_props = MMProps { max_page_size: memory_properties.max_page_size, alignment: memory_properties.alignment };
        let storage = Metal4Storage::new(device.clone(), alignment);
        let mem_manage = MemoryManagement::from_configuration(storage, &mm_props, memory_config.clone());
        // Create a reusable command buffer and allocator ring
        let cb = device.newCommandBuffer();
        let mut allocators: Vec<Retained<ProtocolObject<dyn MTL4CommandAllocator>>> = Vec::new();
        for _ in 0..3 {
            if let Some(a) = device.newCommandAllocator() { allocators.push(a); }
        }
        let frames_in_flight = 3;
        let inflight_staging: Vec<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>> = (0..frames_in_flight).map(|_| Vec::new()).collect();
        let inflight_tables: Vec<Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>> = (0..frames_in_flight).map(|_| Vec::new()).collect();
        let inflight_tensors: Vec<Vec<Retained<ProtocolObject<dyn MTLTensor>>>> = (0..frames_in_flight).map(|_| Vec::new()).collect();
        let inflight_fences: Vec<u64> = vec![0; frames_in_flight];
        // Default to grouped commits (optimal CPU overhead). No debug overrides.
        let defer_commit = true;

        Self {
            logger,
            timing_method,
            memory_properties,
            memory_config,
            mem_manage,
            device,
            queue,
            residency,
            event,
            fence_value: 0,
            cb,
            allocators,
            frames_in_flight,
            frame_cursor: 0,
            pending_cbs: Vec::new(),
            defer_commit,
            inflight_staging,
            inflight_fences,
            inflight_tables,
            inflight_tensors,
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
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut out = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            let size = (desc.shape.iter().product::<usize>() * desc.elem_size) as u64;
            let slice = self.mem_manage.reserve(size)?;
            let handle = Handle::new(slice, None, None, stream_id, 0, size);
            let strides = contiguous_strides(desc.shape);
            out.push(Allocation::new(handle, strides));
        }
        Ok(out)
    }

    fn read<'a>(
        &mut self,
        descriptors: Vec<CopyDescriptor<'a>>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        // Implement using MTLTensor (rank-1 flatten) to align with Metal 4 tensor APIs
        let mut outputs: Vec<Bytes> = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            let elem_count: usize = desc.shape.iter().product();
            let byte_len = elem_count * desc.elem_size;

            // Resolve handle including offsets
            let mut handle = match self.mem_manage.get(desc.binding.memory.clone()) {
                Some(h) => h,
                None => return Box::pin(async { Err(IoError::InvalidHandle) }),
            };
            if let Some(off) = desc.binding.offset_start { handle = handle.offset_start(off); }
            if let Some(off) = desc.binding.offset_end { handle = handle.offset_end(off); }

            // Backing buffer lookup
            let id = handle.id.clone();
            let storage = self.mem_manage.storage();
            let Some(buf) = storage.get_buffer(&id) else {
                return Box::pin(async { Err(IoError::InvalidHandle) });
            };

            // Build a temporary rank-1 tensor view over the buffer slice
            let dims_val: [NSInteger; 1] = [elem_count as NSInteger];
            let strides_val: [NSInteger; 1] = [1 as NSInteger];
            let dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, dims_val.as_ptr()) }.expect("extents");
            let strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, strides_val.as_ptr()) }.expect("strides");
                    let td = MTLTensorDescriptor::new();
                    td.setDimensions(&dims);
                    td.setStrides(Some(&strides));
                    td.setUsage(MTLTensorUsage::Compute);
                    td.setDataType(match desc.elem_size {
                2 => MTLTensorDataType::Float16,
                4 => MTLTensorDataType::Float32,
                1 => MTLTensorDataType::UInt8,
                _ => MTLTensorDataType::Float32,
            });

            let offset = handle.offset() as NSUInteger;
            let tensor: Retained<ProtocolObject<dyn MTLTensor>> = match unsafe { buf.newTensorWithDescriptor_offset_error(&td, offset) } {
                Ok(t) => t,
                Err(_) => return Box::pin(async { Err(IoError::Unknown("Failed to create MTLTensor view".into())) }),
            };

            // Read via getBytes (rank-1 slice covering whole tensor)
            let slice_origin = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, [0 as NSInteger].as_ptr()) }.expect("slice_origin");
            let slice_dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, dims_val.as_ptr()) }.expect("slice_dims");
            let mut host = vec![0u8; byte_len];
            let mut_ptr = core::ptr::NonNull::new(host.as_mut_ptr() as *mut core::ffi::c_void).expect("nonnull");
            unsafe { tensor.getBytes_strides_fromSliceOrigin_sliceDimensions(mut_ptr, &strides, &slice_origin, &slice_dims) };
            outputs.push(Bytes::from_bytes_vec(host));
        }
        Box::pin(async move { Ok(outputs) })
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, &[u8])>,
        _stream_id: StreamId,
    ) -> Result<(), IoError> {
        for (desc, src) in descriptors.into_iter() {
            let elem_count: usize = desc.shape.iter().product();
            let byte_len = elem_count * desc.elem_size;
            if src.len() != byte_len {
                return Err(IoError::Unknown("write: source length mismatch".into()));
            }

            // Resolve handle and buffer
            let mut handle = self
                .mem_manage
                .get(desc.binding.memory.clone())
                .ok_or(IoError::InvalidHandle)?;
            if let Some(off) = desc.binding.offset_start { handle = handle.offset_start(off); }
            if let Some(off) = desc.binding.offset_end { handle = handle.offset_end(off); }

            let id = handle.id.clone();
            let storage = self.mem_manage.storage();
            let Some(buf) = storage.get_buffer(&id) else {
                return Err(IoError::InvalidHandle);
            };

            // Temporary rank-1 tensor view
            let dims_val: [NSInteger; 1] = [elem_count as NSInteger];
            let strides_val: [NSInteger; 1] = [1 as NSInteger];
            let dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, dims_val.as_ptr()) }.expect("extents");
            let strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, strides_val.as_ptr()) }.expect("strides");
            let td = MTLTensorDescriptor::new();
            td.setDimensions(&dims);
            td.setStrides(Some(&strides));
            td.setUsage(MTLTensorUsage::Compute);
            td.setDataType(match desc.elem_size {
                2 => MTLTensorDataType::Float16,
                4 => MTLTensorDataType::Float32,
                1 => MTLTensorDataType::UInt8,
                _ => MTLTensorDataType::Float32,
            });

            let offset = handle.offset() as NSUInteger;
            let tensor: Retained<ProtocolObject<dyn MTLTensor>> = unsafe {
                match buf.newTensorWithDescriptor_offset_error(&td, offset) {
                    Ok(t) => t,
                    Err(_) => return Err(IoError::Unknown("Failed to create MTLTensor view".into())),
                }
            };

            let slice_origin = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, [0 as NSInteger].as_ptr()) }.expect("slice_origin");
            let slice_dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, dims_val.as_ptr()) }.expect("slice_dims");
            let cptr = core::ptr::NonNull::new(src.as_ptr() as *mut core::ffi::c_void).expect("nonnull");
            unsafe { tensor.replaceSliceOrigin_sliceDimensions_withBytes_strides(&slice_origin, &slice_dims, cptr, &strides) };
        }
        Ok(())
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<()> {
        if self.defer_commit && !self.pending_cbs.is_empty() {
            self.flush(_stream_id);
        }
        // Block until the last submitted work (tracked by a shared event) completes.
        let target_value = self.fence_value;
        let event = self.event.clone();
        Box::pin(async move {
            if target_value == 0 {
                return;
            }
            loop {
                let done = event.waitUntilSignaledValue_timeoutMS(target_value, 10);
                if done || event.signaledValue() >= target_value { break; }
            }
        })
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        _stream_id: StreamId,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource> {
        // Map binding -> storage handle -> Metal resource
        let handle = self
            .mem_manage
            .get(binding.memory.clone())
            .expect("Failed to find storage!");
        let handle = match binding.offset_start { Some(off) => handle.offset_start(off), None => handle };
        let handle = match binding.offset_end { Some(off) => handle.offset_end(off), None => handle };
        let resource = self.mem_manage.storage().get(&handle);
        BindingResource::new(binding, resource)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        _stream_id: StreamId,
    ) {
        // Compile to MSL and create pipeline if needed
        let mut compiler = Msl4Compiler::default();
        let compile = kernel.compile(&mut compiler, &Default::default(), kind);

        let mut kid = kernel.id();
        kid.mode(kind);
        // Specialize kernel id by entrypoint, vectorization (line size), and element types
        let fast_math = std::env::var("CUBECL_MTL4_FAST_MATH").ok().as_deref() == Some("1");
        let spec = match compile.repr.as_ref() {
            Some(m) => Msl4SpecKey {
                entry: compile.entrypoint_name.clone(),
                line: m.output_line_size,
                types: m.buffers.iter().map(|p| p.elem).collect(),
                fast_math,
            },
            None => Msl4SpecKey { entry: compile.entrypoint_name.clone(), line: 1, types: alloc::vec::Vec::new(), fast_math },
        };
        let kid = kid.info(spec);

        // Lookup PSO cache by specialized key
        let pso = if let Some(p) = self.pipelines.get(&kid) {
            p.clone()
        } else {
            // Always compile fresh during development to avoid stale PSOs when source changes
            let opts = MTLCompileOptions::new();
            // Ensure MSL 4.0 language features are available
            opts.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version4_0);
            // Heuristic: pick MPP when explicitly named, or when first 3 buffers are (half, half, float)
            let type_mpp = compile
                .repr
                .as_ref()
                .map(|m| m.buffers.as_slice())
                .and_then(|b| {
                    if b.len() >= 3 {
                        let a = b[0].elem; let ha = b[0].has_extended_meta;
                        let bb = b[1].elem; let hb = b[1].has_extended_meta;
                        let c = b[2].elem; let hc = b[2].has_extended_meta;
                        let is_half = matches!(a, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F16)) && matches!(bb, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F16));
                        let is_f32 = matches!(c, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F32));
                        if is_half && is_f32 && ha && hb && hc { Some(true) } else { Some(false) }
                    } else { Some(false) }
                })
                .unwrap_or(false);
            // Explicit switch for MPP conv2d
            let want_mpp_conv = kernel.name().contains("mpp_convolution2d")
                || compile.entrypoint_name.as_str() == "mpp_convolution2d";
            let use_mpp = kernel.name().contains("mpp_matmul_2d")
                || compile.entrypoint_name.as_str() == "mpp_matmul_2d"
                || type_mpp
                || want_mpp_conv;

            // Build source
            let src_text = if want_mpp_conv {
                // Extract dims from metadata for A(0)=NHWC, W(1)=HWIO, D(2)=NHWO; use stride=1, dilation=1, groups=1
                let mut n = 1; let mut h = 1; let mut w = 1; let mut c = 1;
                let mut kh = 1; let mut kw = 1; let mut o = 1;
                let mut hout = 1; let mut wout = 1;
                if let Some(module) = compile.repr.as_ref() {
                    let total = module.buffers.len();
                    let num_ext = module.buffers.iter().filter(|p| p.has_extended_meta).count();
                    let ranks_start = 2 * total;
                    let shape_offs_start = ranks_start + num_ext;
                    let data = &bindings.metadata.data;
                    let mut ext_idx = 0usize;
                    // Activation dims
                    if module.buffers.get(0).map(|p| p.has_extended_meta).unwrap_or(false) {
                        let shape_base = data.get(shape_offs_start + ext_idx).copied().unwrap_or(0) as usize;
                        if shape_base + 3 < data.len() {
                            // NHWC layout expected
                            n = data[shape_base + 0] as i32;
                            h = data[shape_base + 1] as i32;
                            w = data[shape_base + 2] as i32;
                            c = data[shape_base + 3] as i32;
                        }
                        ext_idx += 1;
                    }
                    // Weights dims
                    if module.buffers.get(1).map(|p| p.has_extended_meta).unwrap_or(false) {
                        let shape_base = data.get(shape_offs_start + ext_idx).copied().unwrap_or(0) as usize;
                        if shape_base + 3 < data.len() {
                            kh = data[shape_base + 0] as i32;
                            kw = data[shape_base + 1] as i32;
                            let _ci = data[shape_base + 2] as i32; // input channels
                            o = data[shape_base + 3] as i32; // output channels
                        }
                        ext_idx += 1;
                    }
                    // Destination dims
                    if module.buffers.get(2).map(|p| p.has_extended_meta).unwrap_or(false) {
                        let shape_base = data.get(shape_offs_start + ext_idx).copied().unwrap_or(0) as usize;
                        if shape_base + 3 < data.len() {
                            // NHWO layout expected by spec
                            let _n2 = data[shape_base + 0] as i32;
                            hout = data[shape_base + 1] as i32;
                            wout = data[shape_base + 2] as i32;
                            let _o2 = data[shape_base + 3] as i32;
                        }
                    }
                }
                crate::kernels::mpp_convolution2d_source(n, h, w, c, kh, kw, o, hout, wout, 1, 1, 1, 1, 1)
            } else if use_mpp {
                crate::kernels::mpp_matmul_2d_source()
            } else {
                compile.source.clone()
            };
            #[cfg(debug_assertions)]
            {
                // Lightweight debug: print the generated MSL once per unique kernel
                if !use_mpp {
                    println!("[cubecl-metal4] MSL source for {}:\n{}", compile.entrypoint_name, src_text);
                }
            }
            let src = NSString::from_str(&src_text);
            let library: Retained<ProtocolObject<dyn MTLLibrary>> = self
                .device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .expect("Failed to compile MSL");

            // Strict Metal 4 compiler path: require MTL4Compiler, no alternative compile paths
            let comp_desc = MTL4CompilerDescriptor::new();
            let compiler: Retained<ProtocolObject<dyn MTL4Compiler>> = self
                .device
                .newCompilerWithDescriptor_error(&comp_desc)
                .expect("Metal 4 compiler unavailable. Require macOS with MTL4Compiler support.");

            let lfd = MTL4LibraryFunctionDescriptor::new();
            let entry = if use_mpp { "mpp_matmul_2d" } else { &compile.entrypoint_name };
            let fname = NSString::from_str(entry);
            lfd.setName(Some(&fname));
            lfd.setLibrary(Some(&library));

            let cpdesc = MTL4ComputePipelineDescriptor::new();
            // Upcast to base function descriptor
            let base: &MTL4FunctionDescriptor = lfd.as_super();
            cpdesc.setComputeFunctionDescriptor(Some(base));

            let pso = compiler
                .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&cpdesc, None)
                .expect("Failed to create compute pipeline state with MTL4Compiler");
            // Cache
            self.pipelines.insert(kid.clone(), pso.clone());
            pso
        };

        // For MPP detection below
        let use_mpp = kernel.name().contains("mpp_matmul_2d") || compile.entrypoint_name.as_str() == "mpp_matmul_2d";

        // Create or reuse MTL4 command buffer and begin with next allocator
        let cb = match &self.cb {
            Some(cb) => cb.clone(),
            None => match self.device.newCommandBuffer() { Some(c) => { self.cb = Some(c.clone()); c }, None => return },
        };
        if !self.allocators.is_empty() {
            self.frame_cursor = (self.frame_cursor + 1) % self.allocators.len();
            // Ensure previous work in this frame slot is finished before reusing and dropping old staging buffers
            let prev_fence = self.inflight_fences[self.frame_cursor];
            if prev_fence > 0 {
                let ev = self.event.clone();
                // Busy-wait with short timeout until signaled; keeps API simple here
                loop {
                    let done = ev.waitUntilSignaledValue_timeoutMS(prev_fence, 10);
                    if done || ev.signaledValue() >= prev_fence { break; }
                }
                // Safe to drop prior staging buffers for this frame slot
                self.inflight_staging[self.frame_cursor].clear();
                self.inflight_tables[self.frame_cursor].clear();
                self.inflight_tensors[self.frame_cursor].clear();
                self.inflight_fences[self.frame_cursor] = 0;
            }
            let alloc = &self.allocators[self.frame_cursor];
            let _ = cb.beginCommandBufferWithAllocator(alloc);
        } else if let Some(alloc) = self.device.newCommandAllocator() {
            let _ = cb.beginCommandBufferWithAllocator(&alloc);
        }
        // Ensure residency set is applied to this command buffer
        cb.useResidencySet(&self.residency);
        let Some(encoder) = cb.computeCommandEncoder() else { return; };
        encoder.setComputePipelineState(&pso);

        // Create MTL4 argument table sized to buffer bindings (buffers + metadata + aux + scalars)
        let scalar_bind_count = bindings.scalars.values().map(|_s| 1usize).sum::<usize>();
        let meta_bind_count = if bindings.metadata.data.is_empty() { 0 } else { 1 };
        // Bind aux out_len only when the generated kernel expects it (has_metadata && has an output)
        let want_aux = compile
            .repr
            .as_ref()
            .map(|m| m.has_metadata && m.output_index.is_some())
            .unwrap_or(false);
        let aux_bind_count = if want_aux { 1 } else { 0 };
        let buf_bind_count = bindings.buffers.len() + scalar_bind_count + meta_bind_count + aux_bind_count;
        let at_desc = MTL4ArgumentTableDescriptor::new();
        // Bind buffers + metadata + scalars for all kernels
        let only_buffers = false;
        at_desc.setMaxBufferBindCount(buf_bind_count as NSUInteger);
        if let Ok(arg_table) = self.device.newArgumentTableWithDescriptor_error(&at_desc) {
            // Keep strong references to transient per-dispatch buffers until GPU finishes this frame
            let mut owned: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
            // 1) Bind inputs: bind MTLTensor for typed tensor parameters via resource IDs
            let mut next_index = 0usize;
            let cap = buf_bind_count;
            // Precompute metadata section offsets
            let total_bufs = bindings.buffers.len();
            let data = &bindings.metadata.data;
            let module = compile.repr.as_ref();
            let buf_info = module.map(|m| m.buffers.as_slice());
            // Detect MPP by name or by type signature (half, half -> float with extended metadata)
            let type_mpp = buf_info.and_then(|b| {
                if b.len() >= 3 {
                    let a = b[0].elem; let ha = b[0].has_extended_meta;
                    let bb = b[1].elem; let hb = b[1].has_extended_meta;
                    let c = b[2].elem; let hc = b[2].has_extended_meta;
                    let is_half = matches!(a, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F16)) && matches!(bb, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F16));
                    let is_f32 = matches!(c, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F32));
                    if is_half && is_f32 && ha && hb && hc { Some(true) } else { Some(false) }
                } else { Some(false) }
            }).unwrap_or(false);
            let use_mpp = kernel.name().contains("mpp_matmul_2d") || compile.entrypoint_name.as_str() == "mpp_matmul_2d" || type_mpp;
            let num_ext = buf_info.map(|bi| bi.iter().filter(|p| p.has_extended_meta).count()).unwrap_or(0);
            let ranks_start = 2 * total_bufs;
            let shape_offs_start = ranks_start + num_ext;
            let stride_offs_start = shape_offs_start + num_ext;
            for (i, b) in bindings.buffers.iter().enumerate() {
                let br = self.get_resource(b.clone(), _stream_id);
                let res = br.resource();
                let storage_ref = self.mem_manage.storage();
                let Some(buf) = storage_ref.get_buffer(&res.storage_id) else {
                    eprintln!("[cubecl-metal4] ERROR: Missing underlying MTLBuffer for storage id; aborting binding.");
                    return;
                };
                let info = buf_info.and_then(|bi| bi.get(i));
                let dtype = info.and_then(|p| map_elem_type_to_tensor_dtype(p.elem)).unwrap_or(MTLTensorDataType::UInt8);
                let elem_size = info.map(|p| elem_size_bytes(p.elem)).unwrap_or(1) as u64;
                // For stability, bind as rank-1 tensor view using scalar length even when extended metadata exists.
                // Extended metadata is still passed separately for broadcast/indexing in MSL.
                // Build tensor extents: prefer 2D extents when extended metadata is available,
                // falling back to rank-1 view otherwise.
                let dims;
                let strides;
                if use_mpp {
                    // Compute ext index for this buffer (count of previous extended metas)
                    let mut ext_idx_i = 0usize;
                    if let Some(modref) = module {
                        for k in 0..i { if modref.buffers.get(k).map(|p| p.has_extended_meta).unwrap_or(false) { ext_idx_i += 1; } }
                    }
                    if ext_idx_i < num_ext {
                        // Fetch shape and stride offsets
                        if data.len() > shape_offs_start + ext_idx_i && data.len() > stride_offs_start + ext_idx_i {
                            let shape_base = data[shape_offs_start + ext_idx_i] as usize;
                            let stride_base = data[stride_offs_start + ext_idx_i] as usize;
                            if shape_base < data.len() && stride_base < data.len() {
                                // Derive 2D view even for rank-1: (dim0, 1) with strides (stride0, 1)
                                let d0 = data[shape_base + 0] as NSInteger;
                                let d1 = if shape_base + 1 < data.len() { data[shape_base + 1] as NSInteger } else { 1 as NSInteger };
                                let s0 = data[stride_base + 0] as NSInteger;
                                let s1 = if stride_base + 1 < data.len() { data[stride_base + 1] as NSInteger } else { 1 as NSInteger };
                                // For MPP prefer (cols, rows)
                                let shape_vals = [d1, d0];
                                let stride_vals = [s1, s0];
                                dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 2 as NSUInteger, shape_vals.as_ptr()) }
                                    .expect("extents2");
                                strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 2 as NSUInteger, stride_vals.as_ptr()) }
                                    .expect("strides2");
                            } else {
                                let len: NSInteger = (res.size / elem_size as u64) as NSInteger;
                                let shape_vals = [len];
                                let stride_vals = [1 as NSInteger];
                                dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, shape_vals.as_ptr()) }
                                    .expect("extents");
                                strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, stride_vals.as_ptr()) }
                                    .expect("strides");
                            }
                        } else {
                            let len: NSInteger = (res.size / elem_size as u64) as NSInteger;
                            let shape_vals = [len];
                            let stride_vals = [1 as NSInteger];
                            dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, shape_vals.as_ptr()) }
                                .expect("extents");
                            strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, stride_vals.as_ptr()) }
                                .expect("strides");
                        }
                    } else {
                        let len: NSInteger = (res.size / elem_size as u64) as NSInteger;
                        let shape_vals = [len];
                        let stride_vals = [1 as NSInteger];
                        dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, shape_vals.as_ptr()) }
                            .expect("extents");
                        strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, stride_vals.as_ptr()) }
                            .expect("strides");
                    }
                } else {
                    let len: NSInteger = (res.size / elem_size as u64) as NSInteger;
                    let shape_vals = [len];
                    let stride_vals = [1 as NSInteger];
                    dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, shape_vals.as_ptr()) }
                        .expect("extents");
                    strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, stride_vals.as_ptr()) }
                        .expect("strides");
                }
                if std::env::var("CUBECL_MTL4_DEBUG").ok().as_deref() == Some("1") {
                    let mut dbg_dims: Vec<isize> = Vec::new();
                    let mut dbg_strides: Vec<isize> = Vec::new();
                    // Read back dims/strides from our temporary extents to log (unsafe APIs don't expose readback; log computed arrays instead)
                    // We already have 'shape_vals' and 'stride_vals' in scope above when extended meta is present. For rank-1 we reconstructed arrays.
                    // Here, rebuild basic debug from allocations we computed.
                    // Note: We cannot deref MTLTensorExtents here; just log our computed parameters below instead.
                    let _ = (dbg_dims, dbg_strides); // avoid unused warning
                    println!(
                        "[cubecl-metal4] tensor bind i={} dtype={:?} elem_size={} buffer_size={} offset={} (extended_meta={})",
                        i, dtype, elem_size, res.size, res.offset, info.map(|p| p.has_extended_meta).unwrap_or(false)
                    );
                }
                let td = MTLTensorDescriptor::new();
                td.setDimensions(&dims);
                td.setStrides(Some(&strides));
                td.setUsage(MTLTensorUsage::Compute);
                td.setDataType(dtype);
                let offset = (res.offset as u64) as NSUInteger;
                let tensor = match unsafe { buf.newTensorWithDescriptor_offset_error(&td, offset) } {
                    Ok(t) => t,
                    Err(_) => {
                        eprintln!("[cubecl-metal4] ERROR: Failed to create MTLTensor for slot {}.", next_index);
                        return;
                    }
                };
                let rid = tensor.gpuResourceID();
                unsafe { arg_table.setResource_atBufferIndex(rid, next_index as NSUInteger) };
                // Residency: add underlying buffer allocation when available
                if let Some(underlying) = tensor.buffer() {
                    unsafe {
                        let alloc: &ProtocolObject<dyn MTLAllocation> = core::mem::transmute::<&ProtocolObject<dyn MTLBuffer>, &ProtocolObject<dyn MTLAllocation>>(underlying.as_ref());
                        self.residency.addAllocation(alloc);
                    }
                }
                // Retain tensor view until this frame's fence signals
                self.inflight_tensors[self.frame_cursor].push(tensor);
                next_index += 1;
                if next_index > cap { eprintln!("[cubecl-metal4] ERROR: binding overflows argument table ({} > {})", next_index, cap); return; }
            }
            // Defer residency request until all resources (tensors + metadata + aux + scalars) are bound below

            if !only_buffers {
                // 2) Pack and bind metadata (u32 words)
                // Synthesize base metadata if missing for elementwise kernels
                let mut synthesized: Option<Vec<u32>> = None;
                // Prepare metadata slice; if present, adjust logical lengths to scalar lengths for MSL4 per-lane guards
                let meta_slice: &[u32] = if bindings.metadata.data.is_empty() {
                    if let Some(module) = compile.repr.as_ref() {
                        // base metadata: [buffer_lens[N]] [logical_lengths[N]]
                        let total = bindings.buffers.len();
                        if total > 0 {
                            let mut v: Vec<u32> = Vec::with_capacity(total * 2);
                            // buffer lengths from resource size
                            for (i, bnd) in bindings.buffers.iter().enumerate() {
                                let br = self.get_resource(bnd.clone(), _stream_id);
                                let res = br.resource();
                                let elem = module.buffers.get(i).map(|b| b.elem);
                                let esz = elem.map(elem_size_bytes).unwrap_or(1) as u64;
                            let len = (res.size / esz) as u32;
                                #[cfg(debug_assertions)]
                                println!(
                                    "[cubecl-metal4] synth meta buf#{i}: size={} off={} esz={} -> len={}",
                                    res.size, res.offset, esz, len
                                );
                                v.push(len);
                            }
                            // logical lengths == buffer lengths (best-effort)
                            for i in 0..total {
                                let len = v[i] as u32;
                                v.push(len);
                            }
                            synthesized = Some(v);
                        }
                    }
                    synthesized.as_deref().unwrap_or(&[])
                } else {
                    // Adjust lengths (second block) to scalar element counts based on bound resources
                    let mut meta = bindings.metadata.data.clone();
                    if let Some(module) = compile.repr.as_ref() {
                        let total = bindings.buffers.len();
                        for (i, bnd) in bindings.buffers.iter().enumerate() {
                            let br = self.get_resource(bnd.clone(), _stream_id);
                            let res = br.resource();
                            let elem = module.buffers.get(i).map(|b| b.elem);
                            let esz = elem.map(elem_size_bytes).unwrap_or(1) as u64;
                            let scalar_len = (res.size / esz) as u32;
                            if meta.len() > total + i { meta[total + i] = scalar_len; }
                        }
                    }
                    // Leak adjusted vector for lifetime of dispatch
                    let boxed = meta.into_boxed_slice();
                    let slice: &'static [u32] = Box::leak(boxed);
                    slice
                };

                if !meta_slice.is_empty() {
                    #[cfg(debug_assertions)]
                    {
                        println!("[cubecl-metal4] meta buffer_lens={:?} lengths={:?}", &meta_slice[0..bindings.buffers.len()], &meta_slice[bindings.buffers.len()..(bindings.buffers.len()*2)]);
                    }
                    let meta_bytes: &[u8] = bytemuck::cast_slice(meta_slice);
                    let options = objc2_metal::MTLResourceOptions::StorageModeShared;
                    if let Some(buf) = self.device.newBufferWithLength_options(meta_bytes.len() as NSUInteger, options) {
                        unsafe {
                            let ptr = buf.contents().as_ptr() as *mut u8;
                            core::ptr::copy_nonoverlapping(meta_bytes.as_ptr(), ptr, meta_bytes.len());
                        }
                        let addr = buf.gpuAddress() as MTLGPUAddress;
                        unsafe { arg_table.setAddress_atIndex(addr, next_index as NSUInteger) };
                        unsafe {
                            let alloc: &ProtocolObject<dyn MTLAllocation> = core::mem::transmute::<&ProtocolObject<dyn MTLBuffer>, &ProtocolObject<dyn MTLAllocation>>(buf.as_ref());
                            self.residency.addAllocation(alloc);
                        }
                        owned.push(buf);
                        next_index += 1;
                        if next_index > cap { eprintln!("[cubecl-metal4] ERROR: binding overflows argument table ({} > {})", next_index, cap); return; }
                    } else {
                        return;
                    }
                }

                // 3) Bind auxiliary out_len scalar for elementwise guard
                if let Some(m) = module {
                    if let Some((oi, _)) = m.buffers.iter().enumerate().find(|(_,p)| p.is_writeable) {
                        let br = self.get_resource(bindings.buffers[oi].clone(), _stream_id);
                        let res = br.resource();
                        let elem = m.buffers.get(oi).map(|b| b.elem);
                        let esz = elem.map(elem_size_bytes).unwrap_or(1) as u64;
                        let out_len: u32 = (res.size / esz) as u32;
                        let out_len_arr = [out_len];
                        let aux_bytes: &[u8] = bytemuck::cast_slice(&out_len_arr);
                        let options = objc2_metal::MTLResourceOptions::StorageModeShared;
                        if let Some(buf) = self.device.newBufferWithLength_options(aux_bytes.len() as NSUInteger, options) {
                            unsafe {
                                let nn = buf.contents();
                                let ptr = nn.as_ptr() as *mut u8;
                                core::ptr::copy_nonoverlapping(aux_bytes.as_ptr(), ptr, aux_bytes.len());
                            }
                            let addr2 = buf.gpuAddress() as MTLGPUAddress;
                            unsafe { arg_table.setAddress_atIndex(addr2, next_index as NSUInteger) };
                            unsafe {
                                let alloc: &ProtocolObject<dyn MTLAllocation> = core::mem::transmute::<&ProtocolObject<dyn MTLBuffer>, &ProtocolObject<dyn MTLAllocation>>(buf.as_ref());
                                self.residency.addAllocation(alloc);
                            }
                            owned.push(buf);
                            next_index += 1;
                            if next_index > cap { eprintln!("[cubecl-metal4] ERROR: binding overflows argument table ({} > {})", next_index, cap); return; }
                        }
                    }
                }

                // 4) Pack and bind scalars (u64 units)
                for s in bindings.scalars.values() {
                    let data_u64 = s.data();
                    let scalar_bytes: &[u8] = bytemuck::cast_slice(&data_u64);
                    let h = match self.create_with_data(scalar_bytes, _stream_id) {
                        Ok(h) => h,
                        Err(_) => return,
                    };
                    let br = self.get_resource(h.clone().binding(), _stream_id);
                    let res = br.resource();
                    let addr = (res.gpu_address as usize + res.offset) as MTLGPUAddress;
                    unsafe { arg_table.setAddress_atIndex(addr, next_index as NSUInteger) };
                    // Ensure residency for scalar buffer
                    let storage_ref = self.mem_manage.storage();
                    if let Some(buf) = storage_ref.get_buffer(&res.storage_id) {
                        unsafe {
                            let alloc: &ProtocolObject<dyn MTLAllocation> = core::mem::transmute::<&ProtocolObject<dyn MTLBuffer>, &ProtocolObject<dyn MTLAllocation>>(buf.as_ref());
                            self.residency.addAllocation(alloc);
                        }
                    }
                    next_index += 1;
                    if next_index > cap { eprintln!("[cubecl-metal4] ERROR: binding overflows argument table ({} > {})", next_index, cap); return; }
                }
            }

            // Ensure resources are made resident for this dispatch now that all bindings are in the table
            self.residency.requestResidency();
            self.residency.commit();
            // Optional binding summary (debug)
            if std::env::var("CUBECL_MTL4_DEBUG").ok().as_deref() == Some("1") {
                let scalars = scalar_bind_count;
                let meta = meta_bind_count;
                let aux = aux_bind_count;
                let bufs = bindings.buffers.len();
                println!(
                    "[cubecl-metal4] bind summary: entry={} buffers={} meta={} aux={} scalars={} table_cap={} final_index={}",
                    compile.entrypoint_name,
                    bufs,
                    meta,
                    aux,
                    scalars,
                    buf_bind_count,
                    next_index
                );
            }
            // Attach table after binding is complete (snapshot on dispatch)
            encoder.setArgumentTable(Some(&arg_table));
            // Attach owned buffers to this frame slot to keep them alive until fence signals
            self.inflight_staging[self.frame_cursor].extend(owned.into_iter());
            // Also retain the argument table itself until this frame's fence signals
            self.inflight_tables[self.frame_cursor].push(arg_table);
        }

        // Compute grid and threads
        let mut threads = MTLSize { width: compile.cube_dim.x as usize, height: compile.cube_dim.y as usize, depth: compile.cube_dim.z as usize };
        let mut groups = MTLSize { width: 1, height: 1, depth: 1 };

        // For MPP matmul kernels, infer grid from output tensor shape: grid.x = ceil(N/32), grid.y = ceil(M/64)
        let mpp = {
            let module = compile.repr.as_ref();
            let buf_info = module.map(|m| m.buffers.as_slice());
            let type_mpp = buf_info.and_then(|b| {
                if b.len() >= 3 {
                    let a = b[0].elem; let ha = b[0].has_extended_meta;
                    let bb = b[1].elem; let hb = b[1].has_extended_meta;
                    let c = b[2].elem; let hc = b[2].has_extended_meta;
                    let is_half = matches!(a, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F16)) && matches!(bb, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F16));
                    let is_f32 = matches!(c, cubecl_ir::ElemType::Float(cubecl_ir::FloatKind::F32));
                    if is_half && is_f32 && ha && hb && hc { Some(true) } else { Some(false) }
                } else { Some(false) }
            }).unwrap_or(false);
            kernel.name().contains("mpp_matmul_2d") || compile.entrypoint_name.as_str() == "mpp_matmul_2d" || type_mpp
        };
        if mpp {
            if let Some(module) = compile.repr.as_ref() {
                if module.buffers.len() >= 3 {
                    // Decode shape for buffer index 2 (C)
                    let total_bufs = bindings.buffers.len();
                    let num_ext = module.buffers.iter().filter(|p| p.has_extended_meta).count();
                    let ranks_start = 2 * total_bufs;
                    let shape_offs_start = ranks_start + num_ext;
                    let data = &bindings.metadata.data;
                    // Compute ext_idx for buffer 2
                    let mut ext_idx = 0usize;
                    for k in 0..2 { if module.buffers.get(k).map(|p| p.has_extended_meta).unwrap_or(false) { ext_idx += 1; } }
                    if ext_idx < num_ext && data.len() > shape_offs_start + ext_idx {
                        let shape_base = data[shape_offs_start + ext_idx] as usize;
                        if shape_base + 1 < data.len() {
                            // Shape order is (M, N)
                            let m = data[shape_base + 0] as usize; // rows
                            let n = data[shape_base + 1] as usize; // cols
                            let tile_m = 64usize; let tile_n = 32usize;
                            let gx = (n + tile_n - 1) / tile_n;
                            let gy = (m + tile_m - 1) / tile_m;
                            groups = MTLSize { width: gx.max(1), height: gy.max(1), depth: 1 };
                            // Choose threadgroup size based on pipeline width; aim for 4 SIMD-groups per TG
                            let tew = pso.threadExecutionWidth() as usize;
                            let max_tg = pso.maxTotalThreadsPerThreadgroup() as usize;
                            let want = (tew.max(1)) * 4;
                            let tg_w = want.min(max_tg).max(tew);
                            threads = MTLSize { width: tg_w, height: 1, depth: 1 };
                        }
                    }
                }
            }
        } else {
            // Elementwise kernels (non-MPP): 1D outer work-items equal to ceil(len / line)
            if let Some(module) = compile.repr.as_ref() {
                let total_bufs = bindings.buffers.len();
                // Prefer synthesized meta if present above; otherwise, use provided metadata
                let data_ref: &[u32] = if bindings.metadata.data.is_empty() {
                    // Base metadata synthesized mirrors binding order; rebuild in place for sizing
                    let mut tmp: Vec<u32> = Vec::with_capacity(total_bufs * 2);
                    for (i, bnd) in bindings.buffers.iter().enumerate() {
                        let br = self.get_resource(bnd.clone(), _stream_id);
                        let res = br.resource();
                        let elem = module.buffers.get(i).map(|b| b.elem);
                        let esz = elem.map(elem_size_bytes).unwrap_or(1) as u64;
                        let len = (res.size / esz) as u32;
                        tmp.push(len);
                    }
                    for i in 0..total_bufs { let l = tmp[i]; tmp.push(l); }
                    // Store in a local and then borrow; scope ends at dispatch
                    // To keep borrow live until dispatch, move into a Box
                    let boxed = tmp.into_boxed_slice();
                    let slice: &'static [u32] = Box::leak(boxed);
                    slice
                } else {
                    // Use adjusted metadata as above
                    let mut meta = bindings.metadata.data.clone();
                    let total = bindings.buffers.len();
                    if let Some(module) = compile.repr.as_ref() {
                        for (i, bnd) in bindings.buffers.iter().enumerate() {
                            let br = self.get_resource(bnd.clone(), _stream_id);
                            let res = br.resource();
                            let elem = module.buffers.get(i).map(|b| b.elem);
                            let esz = elem.map(elem_size_bytes).unwrap_or(1) as u64;
                            let scalar_len = (res.size / esz) as u32;
                            if meta.len() > total + i { meta[total + i] = scalar_len; }
                        }
                    }
                    let boxed = meta.into_boxed_slice();
                    let slice: &'static [u32] = Box::leak(boxed);
                    slice
                };
                if let Some((i, _)) = module.buffers.iter().enumerate().find(|(_, p)| p.is_writeable) {
                    let len = if data_ref.len() >= total_bufs * 2 { data_ref[total_bufs + i] as usize } else { 0 };
                    let line = if module.output_line_size > 0 { module.output_line_size as usize } else { 1 };
                    let items = if len > 0 { (len + line - 1) / line } else { 0 };
                    // Choose a simple threadgroup size and grid
                    let tg_w = if items >= 64 { 64 } else if items >= 32 { 32 } else if items >= 16 { 16 } else if items >= 8 { 8 } else { items.max(1) };
                    threads = MTLSize { width: tg_w, height: 1, depth: 1 };
                    groups = MTLSize { width: (items + tg_w - 1) / tg_w, height: 1, depth: 1 };
                }
            }
        }
        // Prefer dispatchThreadgroups; use computed groups/threads
        #[cfg(debug_assertions)]
        println!(
            "[cubecl-metal4] dispatchThreadgroups groups=({}, {}, {}), tg=({}, {}, {})",
            groups.width, groups.height, groups.depth, threads.width, threads.height, threads.depth
        );
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        encoder.endEncoding();
        // Flush residency changes
        self.residency.commit();
        // Finalize CB and either defer or commit
        cb.endCommandBuffer();
        if self.defer_commit {
            self.pending_cbs.push(cb);
        } else {
            let mut one: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; 1] = [
                NonNull::new(Retained::as_ptr(&cb) as *mut _).expect("nonnull"),
            ];
            let arr = NonNull::new(one.as_mut_ptr()).expect("nonnull array ptr");
            unsafe { self.queue.commit_count(arr, 1) };

            // Signal a shared event so the CPU can wait for completion in sync()
            self.fence_value = self.fence_value.saturating_add(1);
            let ev: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.event);
            self.queue.signalEvent_value(ev, self.fence_value);
            // Record fence for this frame slot
            self.inflight_fences[self.frame_cursor] = self.fence_value;
        }
    }

    fn flush(&mut self, _stream_id: StreamId) {
        if self.pending_cbs.is_empty() { return; }
        let mut ptrs: Vec<NonNull<ProtocolObject<dyn MTL4CommandBuffer>>> = Vec::with_capacity(self.pending_cbs.len());
        for cb in &self.pending_cbs {
            if let Some(nn) = NonNull::new(Retained::as_ptr(cb) as *mut _) { ptrs.push(nn); }
        }
        if !ptrs.is_empty() {
            let mut raw = ptrs;
            let arr = NonNull::new(raw.as_mut_ptr()).expect("nonnull array ptr");
            unsafe { self.queue.commit_count(arr, raw.len() as _) };
            self.fence_value = self.fence_value.saturating_add(1);
            let ev: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.event);
            self.queue.signalEvent_value(ev, self.fence_value);
            self.inflight_fences[self.frame_cursor] = self.fence_value;
        }
        self.pending_cbs.clear();
    }

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
fn map_elem_type_to_tensor_dtype(elem: cubecl_ir::ElemType) -> Option<MTLTensorDataType> {
    use cubecl_ir::ElemType::*;
    use cubecl_ir::{FloatKind::*, IntKind::*, UIntKind::*};
    match elem {
        Float(F16) => Some(MTLTensorDataType::Float16),
        Float(BF16) => Some(MTLTensorDataType::BFloat16),
        Float(F32) | Float(TF32) | Float(Flex32) => Some(MTLTensorDataType::Float32),
        Int(I8) => Some(MTLTensorDataType::Int8),
        Int(I16) => Some(MTLTensorDataType::Int16),
        Int(I32) => Some(MTLTensorDataType::Int32),
        UInt(U8) | ElemType::Bool => Some(MTLTensorDataType::UInt8),
        UInt(U16) => Some(MTLTensorDataType::UInt16),
        UInt(U32) => Some(MTLTensorDataType::UInt32),
        // Unsupported directly: return None (not directly mappable)
        _ => None,
    }
}

fn elem_size_bytes(elem: cubecl_ir::ElemType) -> usize {
    use cubecl_ir::ElemType::*;
    use cubecl_ir::{FloatKind::*, IntKind::*, UIntKind::*};
    match elem {
        Float(F16) => 2,
        Float(BF16) => 2,
        Float(F32) | Float(TF32) | Float(Flex32) => 4,
        Float(F64) => 8,
        Int(I8) => 1,
        Int(I16) => 2,
        Int(I32) => 4,
        Int(I64) => 8,
        UInt(U8) | ElemType::Bool => 1,
        UInt(U16) => 2,
        UInt(U32) => 4,
        UInt(U64) => 8,
        // Minifloats packed as u8 currently
        Float(E2M1 | E2M3 | E3M2 | E4M3 | E5M2 | UE8M0) => 1,
    }
}
