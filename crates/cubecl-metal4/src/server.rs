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

use crate::storage::Metal4Storage;
use hashbrown::HashMap;
use objc2::rc::Retained;
use objc2::ClassType;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSInteger, NSUInteger};
use objc2::AnyThread;
use objc2_metal::MTLBuffer;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandBuffer, MTL4CommandQueue,
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
    // May reuse a command buffer in future; currently create per-dispatch
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

        let pso = if let Some(p) = self.pipelines.get(&kid) {
            p.clone()
        } else {
            // Compile source to library for function reference (override for MPP matmul)
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
            let use_mpp = kernel.name().contains("mpp_matmul_2d")
                || compile.entrypoint_name.as_str() == "mpp_matmul_2d"
                || type_mpp;
            let src_text = if use_mpp { crate::kernels::mpp_matmul_2d_source() } else { compile.source.clone() };
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

            // Try Metal 4 compiler path first
            let pso = (|| {
                let comp_desc = MTL4CompilerDescriptor::new();
                let compiler: Retained<ProtocolObject<dyn MTL4Compiler>> = self
                    .device
                    .newCompilerWithDescriptor_error(&comp_desc)
                    .ok()?;

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
                    .ok()?;
                Some(pso)
            })()
            .unwrap_or_else(|| {
                // Fallback to older API if MTL4Compiler path is unavailable
                let entry = if use_mpp { "mpp_matmul_2d" } else { &compile.entrypoint_name };
                let fname = NSString::from_str(entry);
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

        // For MPP detection below
        let use_mpp = kernel.name().contains("mpp_matmul_2d") || compile.entrypoint_name.as_str() == "mpp_matmul_2d";

        // Create MTL4 command buffer and encoder
        let Some(cb) = self.device.newCommandBuffer() else { return; };
        // Begin with an allocator per Metal 4 validation requirements
        if let Some(alloc) = self.device.newCommandAllocator() {
            let _ = cb.beginCommandBufferWithAllocator(&alloc);
        }
        // Ensure residency set is applied to this command buffer
        cb.useResidencySet(&self.residency);
        let Some(encoder) = cb.computeCommandEncoder() else { return; };
        encoder.setComputePipelineState(&pso);

        // Create MTL4 argument table sized to buffer bindings (buffers + metadata + scalars)
        let scalar_bind_count = bindings.scalars.values().map(|_s| 1usize).sum::<usize>();
        let meta_bind_count = if bindings.metadata.data.is_empty() { 0 } else { 1 };
        let buf_bind_count = bindings.buffers.len() + scalar_bind_count + meta_bind_count;
        let at_desc = MTL4ArgumentTableDescriptor::new();
        // For non-MPP elementwise kernels, bind only the user buffers
        let only_buffers = !use_mpp;
        if only_buffers {
            at_desc.setMaxBufferBindCount(bindings.buffers.len() as NSUInteger);
            at_desc.setMaxTextureBindCount(0);
            at_desc.setMaxSamplerStateBindCount(0);
        } else {
            at_desc.setMaxBufferBindCount(buf_bind_count as NSUInteger);
        }
        if let Ok(arg_table) = self.device.newArgumentTableWithDescriptor_error(&at_desc) {
            // 1) Bind inputs: for MPP use tensors via resource IDs; otherwise bind buffer GPU addresses
            let mut next_index = 0usize;
            // Precompute metadata section offsets
            let total_bufs = bindings.buffers.len();
            let data = &bindings.metadata.data;
            let module = compile.repr.as_ref();
            let buf_info = module.map(|m| m.buffers.as_slice());
            let use_mpp = kernel.name().contains("mpp_matmul_2d") || compile.entrypoint_name.as_str() == "mpp_matmul_2d";
            let num_ext = buf_info.map(|bi| bi.iter().filter(|p| p.has_extended_meta).count()).unwrap_or(0);
            let ranks_start = 2 * total_bufs;
            let shape_offs_start = ranks_start + num_ext;
            let stride_offs_start = ranks_start + num_ext * 2;
            let mut ext_seen = 0usize;
            for (i, b) in bindings.buffers.iter().enumerate() {
                let br = self.get_resource(b.clone(), _stream_id);
                let res = br.resource();
                let storage_id = res.storage_id.clone();
                let storage_ref = self.mem_manage.storage();
                if let Some(buf) = storage_ref.get_buffer(&storage_id) {
                    // Build typed tensor descriptor from compiler info + metadata
                    let info = buf_info.and_then(|bi| bi.get(i));
                    let dtype = info.and_then(|p| map_elem_type_to_tensor_dtype(p.elem)).unwrap_or(MTLTensorDataType::UInt8);
                    let elem_size = info.map(|p| elem_size_bytes(p.elem)).unwrap_or(1);
                    // Prefer rank-2 for MPP when extended metadata is present
                    let (rank, dims, strides) = if use_mpp {
                        if let (Some(_p), true) = (info, info.map(|p| p.has_extended_meta).unwrap_or(false)) {
                            if num_ext > 0 && data.len() > stride_offs_start {
                                let ext_idx = ext_seen;
                                ext_seen += 1;
                                let r = data[ranks_start + ext_idx] as usize;
                                let shape_base = data[shape_offs_start + ext_idx] as usize;
                                let stride_base = data[stride_offs_start + ext_idx] as usize;
                                let mut shape_vals = Vec::with_capacity(r);
                                let mut stride_vals = Vec::with_capacity(r);
                                for d in 0..r {
                                    shape_vals.push(data[shape_base + d] as NSInteger);
                                    stride_vals.push(data[stride_base + d] as NSInteger);
                                }
                                let dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), r as NSUInteger, shape_vals.as_ptr()) }.expect("extents");
                                let strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), r as NSUInteger, stride_vals.as_ptr()) }.expect("strides");
                                (r, dims, strides)
                            } else {
                                // Fallback to 1D
                                let len = if data.len() >= total_bufs * 2 { data[total_bufs + i] as NSInteger } else { (b.size() as usize) as NSInteger };
                                let shape_vals = [len];
                                let stride_vals = [1 as NSInteger];
                                let dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, shape_vals.as_ptr()) }.expect("extents");
                                let strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, stride_vals.as_ptr()) }.expect("strides");
                                (1usize, dims, strides)
                            }
                        } else {
                            // No extended meta; 1D fallback
                            let len = if data.len() >= total_bufs * 2 { data[total_bufs + i] as NSInteger } else { ((b.size() as usize) / elem_size) as NSInteger };
                            let shape_vals = [len];
                            let stride_vals = [1 as NSInteger];
                            let dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, shape_vals.as_ptr()) }.expect("extents");
                            let strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, stride_vals.as_ptr()) }.expect("strides");
                            (1usize, dims, strides)
                        }
                    } else {
                        // Default 1D tensor using logical length
                        let len = if data.len() >= total_bufs * 2 { data[total_bufs + i] as NSInteger } else { ((b.size() as usize) / elem_size) as NSInteger };
                        let shape_vals = [len];
                        let stride_vals = [1 as NSInteger];
                        let dims = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, shape_vals.as_ptr()) }.expect("extents");
                        let strides = unsafe { MTLTensorExtents::initWithRank_values(MTLTensorExtents::alloc(), 1 as NSUInteger, stride_vals.as_ptr()) }.expect("strides");
                    (1usize, dims, strides)
                    };
                    let td = MTLTensorDescriptor::new();
                    td.setDimensions(&dims);
                    td.setStrides(Some(&strides));
                    td.setUsage(MTLTensorUsage::Compute);
                    td.setDataType(dtype);
                    let offset = (res.offset as u64) as NSUInteger;
                    #[cfg(debug_assertions)]
                    unsafe {
                        let rank = dims.rank() as usize;
                        let e0 = if rank > 0 { dims.extentAtDimensionIndex(0) } else { 0 };
                        println!(
                            "[cubecl-metal4] tensor slot {} rank={} extent0={} offset={} bytes",
                            next_index, rank, e0, offset
                        );
                    }
                    match unsafe { buf.newTensorWithDescriptor_offset_error(&td, offset) } {
                        Ok(tensor) => {
                            let rid = tensor.gpuResourceID();
                            #[cfg(debug_assertions)]
                            println!("[cubecl-metal4] bound tensor at slot {} (dtype {:?})", next_index, dtype);
                            unsafe { arg_table.setResource_atBufferIndex(rid, next_index as NSUInteger) };
                            // Add tensor and underlying buffer allocations to residency set
                            unsafe {
                                let tensor_alloc: &ProtocolObject<dyn MTLAllocation> = core::mem::transmute::<&ProtocolObject<dyn MTLTensor>, &ProtocolObject<dyn MTLAllocation>>(tensor.as_ref());
                                self.residency.addAllocation(tensor_alloc);
                            }
                            if let Some(underlying) = tensor.buffer() {
                                unsafe {
                                    let alloc: &ProtocolObject<dyn MTLAllocation> = core::mem::transmute::<&ProtocolObject<dyn MTLBuffer>, &ProtocolObject<dyn MTLAllocation>>(underlying.as_ref());
                                    self.residency.addAllocation(alloc);
                                }
                            }
                        }
                        Err(_) => {
                            // If tensor creation fails, fall back to address binding (best effort)
                            let addr = (res.gpu_address as usize + res.offset) as MTLGPUAddress;
                            #[cfg(debug_assertions)]
                            println!("[cubecl-metal4] tensor failed; bound address at slot {}", next_index);
                            unsafe { arg_table.setAddress_atIndex(addr, next_index as NSUInteger) };
                            unsafe {
                                let alloc: &ProtocolObject<dyn MTLAllocation> = core::mem::transmute::<&ProtocolObject<dyn MTLBuffer>, &ProtocolObject<dyn MTLAllocation>>(buf.as_ref());
                                self.residency.addAllocation(alloc);
                            }
                        }
                    }
                } else {
                    let addr = (res.gpu_address as usize + res.offset) as MTLGPUAddress;
                    unsafe { arg_table.setAddress_atIndex(addr, next_index as NSUInteger) };
                }
                next_index += 1;
            }
            // Ensure resources are made resident for this dispatch
            self.residency.requestResidency();
            self.residency.commit();

            if !only_buffers {
                // 2) Pack and bind metadata (u32 words)
                if !bindings.metadata.data.is_empty() {
                    let meta_bytes: &[u8] = bytemuck::cast_slice(&bindings.metadata.data);
                    let meta_handle = match self.create_with_data(meta_bytes, _stream_id) {
                        Ok(h) => h,
                        Err(_) => return,
                    };
                    let br = self.get_resource(meta_handle.clone().binding(), _stream_id);
                    let res = br.resource();
                    let addr = (res.gpu_address as usize + res.offset) as MTLGPUAddress;
                    unsafe { arg_table.setAddress_atIndex(addr, next_index as NSUInteger) };
                    next_index += 1;
                }

                // 3) Pack and bind scalars (u64 units)
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
                    next_index += 1;
                }
            }

            // Attach table after binding is complete (snapshot on dispatch)
            encoder.setArgumentTable(Some(&arg_table));
        }

        // Compute grid and threads
        let (mut gx, mut gy, gz) = match count {
            CubeCount::Static(x, y, z) => (x as usize, y as usize, z as usize),
            CubeCount::Dynamic(_) => (1, 1, 1),
        };
        let mut threads = MTLSize { width: compile.cube_dim.x as usize, height: compile.cube_dim.y as usize, depth: compile.cube_dim.z as usize };
        let mut grid = MTLSize { width: gx, height: gy, depth: gz };
        // For MPP matmul kernels, infer grid from output tensor shape: grid.x = ceil(N/32), grid.y = ceil(M/64)
        let mpp = kernel.name().contains("mpp_matmul_2d") || compile.entrypoint_name.as_str() == "mpp_matmul_2d";
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
                    for k in 0..2 {
                        if module.buffers.get(k).map(|p| p.has_extended_meta).unwrap_or(false) { ext_idx += 1; }
                    }
                    if ext_idx < num_ext && data.len() > shape_offs_start + ext_idx {
                        let shape_base = data[shape_offs_start + ext_idx] as usize;
                        if shape_base + 1 < data.len() {
                            let n = data[shape_base + 0] as usize; // innermost
                            let m = data[shape_base + 1] as usize; // outer
                            let tile_m = 64usize; let tile_n = 32usize;
                            gx = (n + tile_n - 1) / tile_n;
                            gy = (m + tile_m - 1) / tile_m;
                            grid = MTLSize { width: gx, height: gy, depth: 1 };
                            threads = MTLSize { width: 1, height: 1, depth: 1 };
                        }
                    }
                }
            }
        }
        // For elementwise kernels (non-MPP), default to 1D grid sized to first writeable buffer length
        if !mpp {
            if let Some(module) = compile.repr.as_ref() {
                let total_bufs = bindings.buffers.len();
                let data = &bindings.metadata.data;
                // Find first writeable buffer index
                if let Some((i, _)) = module.buffers.iter().enumerate().find(|(_, p)| p.is_writeable) {
                    let len = if data.len() >= total_bufs * 2 { data[total_bufs + i] as usize } else { 0 };
                    if len > 0 {
                        gx = len; gy = 1;
                        grid = MTLSize { width: gx, height: gy, depth: 1 };
                        threads = MTLSize { width: 1, height: 1, depth: 1 };
                    }
                }
            }
        }
        // Prefer dispatchThreadgroups; choose a simple threadgroup size and grid
        let tg_w = if gx >= 64 { 64 } else if gx >= 32 { 32 } else if gx >= 16 { 16 } else if gx >= 8 { 8 } else { gx.max(1) };
        let threadgroup = MTLSize { width: tg_w, height: 1, depth: 1 };
        let groups = MTLSize { width: (gx + tg_w - 1) / tg_w, height: gy, depth: gz };
        #[cfg(debug_assertions)]
        println!(
            "[cubecl-metal4] dispatchThreadgroups groups=({}, {}, {}), tg=({}, {}, {})",
            groups.width, groups.height, groups.depth, threadgroup.width, threadgroup.height, threadgroup.depth
        );
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threadgroup);
        encoder.endEncoding();
        // Flush residency changes
        self.residency.commit();
        // Finalize CB and commit via MTL4CommandQueue
        cb.endCommandBuffer();
        let mut one: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; 1] = [
            NonNull::new(Retained::as_ptr(&cb) as *mut _).expect("nonnull"),
        ];
        let arr = NonNull::new(one.as_mut_ptr()).expect("nonnull array ptr");
        unsafe { self.queue.commit_count(arr, 1) };

        // Signal a shared event so the CPU can wait for completion in sync()
        self.fence_value = self.fence_value.saturating_add(1);
        let ev: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.event);
        self.queue.signalEvent_value(ev, self.fence_value);
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
        // Unsupported directly: fallback
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
