use cubecl_common::device::{Device, DeviceId};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct Metal4Device;

impl Device for Metal4Device {
    fn from_id(_device_id: DeviceId) -> Self {
        Metal4Device
    }

    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, 0)
    }

    fn device_count(_type_id: u16) -> usize {
        // Metal 4 on Apple Silicon: typically a single device for compute.
        1
    }
}

