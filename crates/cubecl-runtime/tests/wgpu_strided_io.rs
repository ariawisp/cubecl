#![cfg(all(feature = "wgpu", feature = "std"))]

use cubecl_common::{future::block_on, reader::read_sync};
use cubecl_runtime::server::CopyDescriptor;

#[test]
fn wgpu_strided_io_roundtrip_u8_rows_pitched_rank2() {
    type R = cubecl_wgpu::WgpuRuntime;
    let client = R::client(&R::Device::default());

    let rows: usize = 4;
    let cols: usize = 5;
    let pitch_elems: usize = 8; // >= cols; introduces padding per row
    let elem_size: usize = 1; // u8
    let total_bytes = rows * pitch_elems * elem_size;

    let handle = client.empty(total_bytes);

    let mut data = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] = (r as u8) * 32 + (c as u8);
        }
    }

    let binding = handle.clone().binding();
    let shape = [rows, cols];
    let strides = [pitch_elems, 1];

    let write_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    block_on(client.write_async(vec![(write_desc, &data)])).expect("pitched write ok");

    let read_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    let out = read_sync(client.read_tensor_async(vec![read_desc]));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], data);
}

#[test]
fn wgpu_strided_io_roundtrip_f32_rows_pitched_rank2() {
    type R = cubecl_wgpu::WgpuRuntime;
    let client = R::client(&R::Device::default());

    let rows: usize = 3;
    let cols: usize = 7;
    let pitch_elems: usize = 10; // >= cols; introduces padding per row
    let elem_size: usize = core::mem::size_of::<f32>();
    let total_bytes = rows * pitch_elems * elem_size;

    let handle = client.empty(total_bytes);

    let mut data = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] = (r as f32) * 100.0 + (c as f32);
        }
    }
    let bytes: &[u8] = bytemuck::cast_slice(&data);

    let binding = handle.clone().binding();
    let shape = [rows, cols];
    let strides = [pitch_elems, 1];

    let write_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    block_on(client.write_async(vec![(write_desc, bytes)])).expect("pitched write ok");

    let read_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    let out = read_sync(client.read_tensor_async(vec![read_desc]));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], bytes);
}

#[test]
fn wgpu_strided_io_roundtrip_rank3_inner_contiguous() {
    type R = cubecl_wgpu::WgpuRuntime;
    let client = R::client(&R::Device::default());

    let d0 = 2usize;
    let d1 = 3usize;
    let d2 = 4usize; // inner contiguous
    let rows = d0 * d1;
    let cols = d2;
    let pitch_elems = 8; // >= cols
    let elem_size = 1usize;
    let total_bytes = rows * pitch_elems * elem_size;

    let handle = client.empty(total_bytes);

    let mut data = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] = ((r * 13 + c) % 251) as u8;
        }
    }

    let binding = handle.clone().binding();
    let shape = [d0, d1, d2];
    let strides = [d1 * pitch_elems, pitch_elems, 1];

    let write_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    block_on(client.write_async(vec![(write_desc, &data)])).expect("pitched write ok");

    let read_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    let out = read_sync(client.read_tensor_async(vec![read_desc]));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], data);
}
