extern crate alloc;
use alloc::sync::Arc;
use cubecl_common::future;
use cubecl_core::{CubeCount, CubeDim, Runtime, ir::TargetProperties};
use cubecl_runtime::channel;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::DeviceProperties;
use cubecl_runtime::memory_management::{HardwareProperties, MemoryDeviceProperties};
use cubecl_runtime::{ComputeRuntime, logging::ServerLogger};

use crate::device::Metal4Device;
use crate::server::Metal4Server;
use cubecl_msl4::Msl4Compiler;

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
            // Build minimal DeviceProperties; tune later for actual Metal 4 limits
            let mem_props = MemoryDeviceProperties {
                max_page_size: 1 << 30, // placeholder
                alignment: 256,
            };
            let hardware_props = HardwareProperties {
                plane_size_min: 32,
                plane_size_max: 32,
                max_bindings: 64,
                max_shared_memory_size: 64 * 1024,
                max_cube_count: CubeCount::new_3d(65535, 65535, 65535),
                max_units_per_cube: 1024,
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
