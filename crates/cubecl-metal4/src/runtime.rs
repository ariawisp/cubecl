extern crate alloc;
use alloc::sync::Arc;
use cubecl_core::{CubeCount, CubeDim, Runtime, ir::TargetProperties};
use cubecl_runtime::channel;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::DeviceProperties;
use cubecl_runtime::memory_management::{HardwareProperties, MemoryDeviceProperties};
use cubecl_runtime::{ComputeRuntime, logging::ServerLogger};

use crate::device::Metal4Device;
use crate::server::Metal4Server;
use crate::compiler::Msl4Compiler;

// Probe device/pipeline limits via Metal 4 APIs
use objc2::rc::Retained;
use objc2::ClassType;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::MTLDevice as _;
use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor, MTL4FunctionDescriptor,
    MTL4LibraryFunctionDescriptor, MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLLanguageVersion, MTLLibrary,
};

fn probe_pipeline_limits() -> Option<(usize, usize)> {
    // Returns (thread_execution_width, max_total_threads_per_tg)
    let device = MTLCreateSystemDefaultDevice()?;
    // Minimal MSL 4.0 kernel
    let src = r#"#include <metal_stdlib>
using namespace metal;
kernel void __probe(uint3 tid [[thread_position_in_grid]]) { (void)tid; }"#;
    let src_ns = NSString::from_str(src);

    let opts = MTLCompileOptions::new();
    opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library: Retained<ProtocolObject<dyn MTLLibrary>> = device
        .newLibraryWithSource_options_error(&src_ns, Some(&opts))
        .ok()?;

    let comp_desc = MTL4CompilerDescriptor::new();
    let compiler: Retained<ProtocolObject<dyn MTL4Compiler>> = device
        .newCompilerWithDescriptor_error(&comp_desc)
        .ok()?;

    let lf = MTL4LibraryFunctionDescriptor::new();
    let fname = NSString::from_str("__probe");
    lf.setName(Some(&fname));
    lf.setLibrary(Some(&library));

    let cpdesc = MTL4ComputePipelineDescriptor::new();
    let base: &MTL4FunctionDescriptor = lf.as_super();
    cpdesc.setComputeFunctionDescriptor(Some(base));
    let pso: Retained<ProtocolObject<dyn MTLComputePipelineState>> = compiler
        .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&cpdesc, None)
        .ok()?;

    let tew = pso.threadExecutionWidth() as usize;
    let max_tg = pso.maxTotalThreadsPerThreadgroup() as usize;
    Some((tew.max(1), max_tg.max(tew)))
}

#[derive(Debug)]
pub struct Metal4Runtime;

type Server = Metal4Server;
type Channel = channel::MutexComputeChannel<Server>;

static RUNTIME: ComputeRuntime<Metal4Device, Server, Channel> = ComputeRuntime::new();

impl Runtime for Metal4Runtime {
    type Compiler = Msl4Compiler;
    type Server = Metal4Server;
    type Channel = Channel;
    type Device = Metal4Device;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, || {
            // Build DeviceProperties from device/pipeline probing when available
            let mem_props = MemoryDeviceProperties {
                max_page_size: 1 << 30, // placeholder
                alignment: 256,
            };
            let (_tew, max_tg) = probe_pipeline_limits().unwrap_or((32, 1024));
            let hardware_props = HardwareProperties {
                plane_size_min: 32,
                plane_size_max: 32,
                max_bindings: 64,
                max_shared_memory_size: 64 * 1024, // conservative default
                max_cube_count: CubeCount::new_3d(65535, 65535, 65535),
                max_units_per_cube: max_tg as u32,
                // Favor generous XY with conservative Z; TEW informs optimal widths
                max_cube_dim: CubeDim::new_3d(1024, 1024, 64),
                num_streaming_multiprocessors: None,
                num_tensor_cores: None,
                min_tensor_cores_dim: None,
            };
            let timing_method = cubecl_common::profile::TimingMethod::System;
            let dev_props = DeviceProperties::new(Default::default(), mem_props.clone(), hardware_props, timing_method);

            let server = Metal4Server::new(
                mem_props,
                cubecl_core::MemoryConfiguration::default(),
                timing_method,
                Arc::new(ServerLogger::default()),
            );
            let channel = Channel::new(server);
            ComputeClient::new(channel, dev_props, ("metal4",))
        })
    }

    fn name(_client: &ComputeClient<Self::Server, Self::Channel>) -> &'static str {
        "metal4<msl4>"
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[8, 4, 2, 1]
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u16::MAX as u32, u16::MAX as u32, u16::MAX as u32)
    }

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool {
        if shape.is_empty() {
            return true;
        }
        // Compute contiguous strides and compare
        let mut expected = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            expected[i] = expected[i + 1] * shape[i + 1];
        }
        expected.into_iter().zip(strides).all(|(e, &s)| e == s)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: Default::default(),
        }
    }
}
