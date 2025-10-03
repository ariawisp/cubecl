use cubecl_runtime::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId};
use hashbrown::HashMap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLBuffer, MTLDevice as _};

#[derive(Clone, Debug, Copy)]
pub struct Metal4Resource {
    pub size: u64,
    pub offset: usize,
    pub storage_id: StorageId,
    pub gpu_address: u64,
}

#[derive(Debug)]
pub struct Metal4Storage {
    device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    alignment: usize,
    buffers: HashMap<StorageId, Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Metal4Storage {
    pub fn new(device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>, alignment: usize) -> Self {
        Self { device, alignment, buffers: HashMap::new() }
    }

    pub fn get_buffer(&self, id: &StorageId) -> Option<&Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.buffers.get(id)
    }
}

// Metal objects are thread-safe to reference; we mark storage Send/Sync for runtime usage.
unsafe impl Send for Metal4Storage {}
unsafe impl Sync for Metal4Storage {}

impl ComputeStorage for Metal4Storage {
    type Resource = Metal4Resource;

    fn alignment(&self) -> usize { self.alignment }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let id = handle.id.clone();
        let addr = self
            .buffers
            .get(&id)
            .map(|b| b.gpuAddress() as u64)
            .unwrap_or(0);
        Metal4Resource { size: handle.size(), offset: handle.offset() as usize, storage_id: id, gpu_address: addr }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        use cubecl_runtime::storage::{StorageHandle as SH, StorageUtilization};
        let id = StorageId::new();
        let options = objc2_metal::MTLResourceOptions::StorageModeShared;
        let buf = self.device.newBufferWithLength_options(size as NSUInteger, options)
            .ok_or_else(|| IoError::Unknown("Failed to create MTLBuffer".into()))?;
        self.buffers.insert(id, buf);
        let utilization = StorageUtilization { offset: 0, size };
        Ok(SH::new(id, utilization))
    }

    fn dealloc(&mut self, id: StorageId) { let _ = self.buffers.remove(&id); }
}
