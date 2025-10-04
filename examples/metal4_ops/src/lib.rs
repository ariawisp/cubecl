use cubecl::prelude::*;
use cubecl::future;

#[cube(launch_unchecked)]
fn add_array<F: Float>(a: &Array<Line<F>>, b: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
fn sub_array<F: Float>(a: &Array<Line<F>>, b: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] - b[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
fn mul_array<F: Float>(a: &Array<Line<F>>, b: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
fn div_array<F: Float>(a: &Array<Line<F>>, b: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] / b[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
fn neg_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() {
        out[ABSOLUTE_POS] = -a[ABSOLUTE_POS];
    }
}

pub fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let a = &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let b_broadcast = &[2.0f32];
    let n = a.len();
    let vec = 4u8; // vectorization factor

    let out_add = client.empty(n * core::mem::size_of::<f32>());
    let out_sub = client.empty(n * core::mem::size_of::<f32>());
    let out_mul = client.empty(n * core::mem::size_of::<f32>());
    let out_div = client.empty(n * core::mem::size_of::<f32>());
    let out_neg = client.empty(n * core::mem::size_of::<f32>());
    let out_add_bcast = client.empty(n * core::mem::size_of::<f32>());

    let a_handle = client.create(f32::as_bytes(a));
    let b_handle = client.create(f32::as_bytes(b));
    let b_bcast_handle = client.create(f32::as_bytes(b_broadcast));

    unsafe {
        add_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1),
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&b_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&out_add, n, vec),
        );
        sub_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1),
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&b_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&out_sub, n, vec),
        );
        mul_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1),
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&b_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&out_mul, n, vec),
        );
        div_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1),
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&b_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&out_div, n, vec),
        );
        neg_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1),
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&out_neg, n, vec),
        );
        // add broadcast: b is length 1
        add_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1),
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, vec),
            ArrayArg::from_raw_parts::<f32>(&b_bcast_handle, 1, vec),
            ArrayArg::from_raw_parts::<f32>(&out_add_bcast, n, vec),
        );
    }

    future::block_on(client.sync());

    let add_b = client.read_one(out_add);
    let sub_b = client.read_one(out_sub);
    let mul_b = client.read_one(out_mul);
    let div_b = client.read_one(out_div);
    let neg_b = client.read_one(out_neg);
    let add_bcast_b = client.read_one(out_add_bcast);
    let add = f32::from_bytes(&add_b);
    let sub = f32::from_bytes(&sub_b);
    let mul = f32::from_bytes(&mul_b);
    let div = f32::from_bytes(&div_b);
    let neg = f32::from_bytes(&neg_b);
    let add_bcast = f32::from_bytes(&add_bcast_b);
    // CPU expected
    let exp_add: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| x + y).collect();
    let exp_sub: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| x - y).collect();
    let exp_mul: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| x * y).collect();
    let exp_div: Vec<f32> = a.iter().zip(b.iter()).map(|(x,y)| x / y).collect();
    let exp_neg: Vec<f32> = a.iter().map(|x| -*x).collect();
    let exp_add_bcast: Vec<f32> = a.iter().map(|x| *x + b_broadcast[0]).collect();

    let approx_eq = |u:&[f32], v:&[f32]| -> bool {
        if u.len()!=v.len() { return false; }
        u.iter().zip(v.iter()).all(|(x,y)| (x-y).abs() < 1e-5*f32::max(1.0, y.abs()))
    };
    println!("add ok? {}", approx_eq(&add, &exp_add));
    println!("sub ok? {}", approx_eq(&sub, &exp_sub));
    println!("mul ok? {}", approx_eq(&mul, &exp_mul));
    println!("div ok? {}", approx_eq(&div, &exp_div));
    println!("neg ok? {}", approx_eq(&neg, &exp_neg));
    println!("add(bcast) ok? {}", approx_eq(&add_bcast, &exp_add_bcast));
}
