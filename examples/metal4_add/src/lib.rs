use cubecl::prelude::*;
use cubecl::future;

#[cube(launch_unchecked)]
fn add_array<F: Float>(a: &Array<Line<F>>, b: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let a = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let n = a.len();
    let vec = 1u8;

    let out_handle = client.empty(n * core::mem::size_of::<f32>());
    let a_handle = client.create(f32::as_bytes(a));
    let b_handle = client.create(f32::as_bytes(b));

    unsafe {
        add_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new((n as u32 + vec as u32 - 1) / vec as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&b_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&out_handle, n, vec),
        )
    }

    // Ensure GPU work completed before reading back
    future::block_on(client.sync());

    let bytes = client.read_one(out_handle);
    let out = f32::from_bytes(&bytes);
    println!("Metal4 add: {:?}", out);
}
