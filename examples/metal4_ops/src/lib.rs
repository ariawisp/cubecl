use cubecl::prelude::*;
use cubecl::future;
#[cfg(feature = "metal4")]
use half::f16;

// Helper trait to convert scalar values to f32 for approximate comparisons
trait ToF32 { fn to_f32(self) -> f32; }
impl ToF32 for f32 { fn to_f32(self) -> f32 { self } }
#[cfg(feature = "metal4")]
impl ToF32 for f16 { fn to_f32(self) -> f32 { f32::from(self) } }

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
    // Helper to run float kernels for a type and vectorization
    fn run_float_one<R: Runtime, F: Float + CubeElement + ToF32>(client: &ComputeClient<R::Server, R::Channel>, vec: u8) {
        let a: Vec<F> = (1..=9).map(|v| F::new(v as f32)).collect();
        let b: Vec<F> = (1..=9).map(|v| F::new(10.0 * (v as f32))).collect();
        let b_broadcast: Vec<F> = vec![F::new(2.0)];
        let n = a.len();

        let out_add = client.empty(n * core::mem::size_of::<F>());
        let out_sub = client.empty(n * core::mem::size_of::<F>());
        let out_mul = client.empty(n * core::mem::size_of::<F>());
        let out_div = client.empty(n * core::mem::size_of::<F>());
        let out_neg = client.empty(n * core::mem::size_of::<F>());
        let out_add_bcast = client.empty(n * core::mem::size_of::<F>());

        let a_handle = client.create(F::as_bytes(&a));
        let b_handle = client.create(F::as_bytes(&b));
        let b_bcast_handle = client.create(F::as_bytes(&b_broadcast));

        unsafe {
            let dim = CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1);
            add_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_add, n, vec));
            sub_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_sub, n, vec));
            mul_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_mul, n, vec));
            div_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_div, n, vec));
            neg_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_neg, n, vec));
            // add broadcast: b is length 1
            add_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&b_bcast_handle, 1, vec),
                ArrayArg::from_raw_parts::<F>(&out_add_bcast, n, vec));
        }

        future::block_on(client.sync());

        let add_b = client.read_one(out_add);
        let sub_b = client.read_one(out_sub);
        let mul_b = client.read_one(out_mul);
        let div_b = client.read_one(out_div);
        let neg_b = client.read_one(out_neg);
        let add_bcast_b = client.read_one(out_add_bcast);
        let add = F::from_bytes(&add_b);
        let sub = F::from_bytes(&sub_b);
        let mul = F::from_bytes(&mul_b);
        let div = F::from_bytes(&div_b);
        let neg = F::from_bytes(&neg_b);
        let add_bcast = F::from_bytes(&add_bcast_b);

        let exp_add: Vec<F> = a.iter().zip(b.iter()).map(|(x,y)| *x + *y).collect();
        let exp_sub: Vec<F> = a.iter().zip(b.iter()).map(|(x,y)| *x - *y).collect();
        let exp_mul: Vec<F> = a.iter().zip(b.iter()).map(|(x,y)| *x * *y).collect();
        let exp_div: Vec<F> = a.iter().zip(b.iter()).map(|(x,y)| *x / *y).collect();
        let exp_neg: Vec<F> = a.iter().map(|x| -*x).collect();
        let exp_add_bcast: Vec<F> = a.iter().map(|x| *x + b_broadcast[0]).collect();

        let approx_eq = |u:&[F], v:&[F]| -> bool {
            if u.len() != v.len() { return false; }
            u.iter().zip(v.iter()).all(|(x,y)| {
                let ux: f32 = (*x).to_f32();
                let vy: f32 = (*y).to_f32();
                (ux - vy).abs() < 1e-3 * f32::max(1.0, vy.abs())
            })
        };
        println!("[float {:?} vec={}] add={} sub={} mul={} div={} neg={} add(bcast)={}",
                 core::any::type_name::<F>(), vec,
                 approx_eq(&add, &exp_add), approx_eq(&sub, &exp_sub),
                 approx_eq(&mul, &exp_mul), approx_eq(&div, &exp_div),
                 approx_eq(&neg, &exp_neg), approx_eq(&add_bcast, &exp_add_bcast));
    }

    // Integer path (no neg/broadcast)
    #[cube(launch_unchecked)]
    fn add_array_n<N: Numeric>(a: &Array<Line<N>>, b: &Array<Line<N>>, out: &mut Array<Line<N>>) {
        if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS]; }
    }
    #[cube(launch_unchecked)]
    fn sub_array_n<N: Numeric>(a: &Array<Line<N>>, b: &Array<Line<N>>, out: &mut Array<Line<N>>) {
        if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = a[ABSOLUTE_POS] - b[ABSOLUTE_POS]; }
    }
    #[cube(launch_unchecked)]
    fn mul_array_n<N: Numeric>(a: &Array<Line<N>>, b: &Array<Line<N>>, out: &mut Array<Line<N>>) {
        if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS]; }
    }
    #[cube(launch_unchecked)]
    fn div_array_n<N: Numeric>(a: &Array<Line<N>>, b: &Array<Line<N>>, out: &mut Array<Line<N>>) {
        if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = a[ABSOLUTE_POS] / b[ABSOLUTE_POS]; }
    }

    fn run_int_one<
        R: Runtime,
        N: Numeric
            + CubeElement
            + num_traits::PrimInt
            + num_traits::ops::wrapping::WrappingAdd
            + num_traits::ops::wrapping::WrappingSub
            + num_traits::ops::wrapping::WrappingMul,
    >(client: &ComputeClient<R::Server, R::Channel>, vec: u8) {
        let a: Vec<N> = (1..=9).map(|v| N::from_int(v as i64)).collect();
        let b: Vec<N> = (1..=9).map(|v| N::from_int(10 * v as i64)).collect();
        let n = a.len();

        let out_add = client.empty(n * core::mem::size_of::<N>());
        let out_sub = client.empty(n * core::mem::size_of::<N>());
        let out_mul = client.empty(n * core::mem::size_of::<N>());
        let out_div = client.empty(n * core::mem::size_of::<N>());

        let a_handle = client.create(N::as_bytes(&a));
        let b_handle = client.create(N::as_bytes(&b));

        unsafe {
            let dim = CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1);
            add_array_n::launch_unchecked::<N, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<N>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<N>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<N>(&out_add, n, vec));
            sub_array_n::launch_unchecked::<N, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<N>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<N>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<N>(&out_sub, n, vec));
            mul_array_n::launch_unchecked::<N, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<N>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<N>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<N>(&out_mul, n, vec));
            div_array_n::launch_unchecked::<N, R>(&client, CubeCount::Static(1, 1, 1), dim,
                ArrayArg::from_raw_parts::<N>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<N>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<N>(&out_div, n, vec));
        }

        future::block_on(client.sync());

        let add_b = client.read_one(out_add);
        let sub_b = client.read_one(out_sub);
        let mul_b = client.read_one(out_mul);
        let div_b = client.read_one(out_div);
        let add = N::from_bytes(&add_b);
        let sub = N::from_bytes(&sub_b);
        let mul = N::from_bytes(&mul_b);
        let div = N::from_bytes(&div_b);

        let exp_add: Vec<N> = a.iter().zip(b.iter()).map(|(x,y)| x.wrapping_add(y)).collect();
        let exp_sub: Vec<N> = a.iter().zip(b.iter()).map(|(x,y)| x.wrapping_sub(y)).collect();
        let exp_mul: Vec<N> = a.iter().zip(b.iter()).map(|(x,y)| x.wrapping_mul(y)).collect();
        let exp_div: Vec<N> = a.iter().zip(b.iter()).map(|(x,y)| *x / *y).collect();

        let equal = |u:&[N], v:&[N]| -> bool { u == v };
        println!("[int {:?} vec={}] add={} sub={} mul={} div={}",
                 core::any::type_name::<N>(), vec,
                 equal(&add, &exp_add), equal(&sub, &exp_sub), equal(&mul, &exp_mul), equal(&div, &exp_div));
    }

    // Sweep vectorization and types
    for &vec in &[1u8, 2, 4, 8] {
        run_float_one::<R, f32>(&client, vec);
        run_int_one::<R, u8>(&client, vec);
        run_int_one::<R, i32>(&client, vec);
        #[cfg(feature = "metal4")]
        run_float_one::<R, f16>(&client, vec);
    }

    // Rank-2 broadcast demo: A shape (M,N), B shape (M,1)
    run_rank2_broadcast::<R, f32>(&client, 3, 4, 4);
    // Rank-2 broadcast demo: A shape (1,N), B shape (M,N)
    run_rank2_broadcast_row::<R, f32>(&client, 3, 5, 4);
}

#[cube(launch_unchecked)]
fn add_tensor_2d<F: Float>(a: &Tensor<Line<F>>, b: &Tensor<Line<F>>, out: &mut Tensor<Line<F>>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

fn compact_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() { return vec![]; }
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len()-1).rev() { s[i] = s[i+1]*shape[i+1]; }
    s
}

fn run_rank2_broadcast<R: Runtime, F: Float + CubeElement + ToF32>(client: &ComputeClient<R::Server, R::Channel>, m: usize, n: usize, vec: u8) {
    let shape_out = vec![m, n];
    let shape_b = vec![m, 1];
    let strides_out = compact_strides(&shape_out);
    let strides_b = compact_strides(&shape_b);
    let elems = m * n;

    // Create input data
    let mut a: Vec<F> = Vec::with_capacity(elems);
    let mut b: Vec<F> = Vec::with_capacity(m);
    for i in 0..m { for j in 0..n { a.push(F::new((i as f32) * 10.0 + j as f32)); } }
    for i in 0..m { b.push(F::new((i as f32) * 1.0)); }
    // Expand b to shape (m,1)
    let mut b_full: Vec<F> = Vec::with_capacity(m);
    for i in 0..m { b_full.push(b[i]); }

    let a_h = client.create(F::as_bytes(&a));
    let b_h = client.create(F::as_bytes(&b_full));
    let out_h = client.empty(elems * core::mem::size_of::<F>());

    unsafe {
        let dim = CubeDim::new(std::cmp::max(((elems as u32) + vec as u32 - 1) / vec as u32, 1), 1, 1);
        add_tensor_2d::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(1,1,1),
            dim,
            TensorArg::from_raw_parts::<F>(&a_h, &strides_out, &shape_out, vec),
            TensorArg::from_raw_parts::<F>(&b_h, &strides_b, &shape_b, vec),
            TensorArg::from_raw_parts::<F>(&out_h, &strides_out, &shape_out, vec),
        );
    }
    future::block_on(client.sync());

    let out_b = client.read_one(out_h);
    let out = F::from_bytes(&out_b);
    // Expected CPU result with broadcast along N
    let mut expect: Vec<F> = Vec::with_capacity(elems);
    for i in 0..m { for j in 0..n { expect.push(F::new(((i as f32) * 10.0 + j as f32) + (i as f32))); } }
    let approx = |u:&[F], v:&[F]| -> bool {
        if u.len()!=v.len() { return false; }
        u.iter().zip(v.iter()).all(|(x,y)| {
            let ux: f32 = (*x).to_f32();
            let vy: f32 = (*y).to_f32();
            (ux - vy).abs() < 1e-3 * f32::max(1.0, vy.abs())
        })
    };
    println!("[rank2 f32 vec={}] add(MxN + Mx1) ok? {}", vec, approx(out, &expect));
}

fn run_rank2_broadcast_row<R: Runtime, F: Float + CubeElement + ToF32>(client: &ComputeClient<R::Server, R::Channel>, m: usize, n: usize, vec: u8) {
    let shape_out = vec![m, n];
    let shape_a = vec![1, n];
    let strides_out = compact_strides(&shape_out);
    let strides_a = compact_strides(&shape_a);
    let elems = m * n;

    // Create input data
    let mut a_row: Vec<F> = Vec::with_capacity(n);
    let mut b: Vec<F> = Vec::with_capacity(elems);
    for j in 0..n { a_row.push(F::new(j as f32)); }
    for i in 0..m { for j in 0..n { b.push(F::new((i as f32) * 2.0 + j as f32)); } }

    let a_h = client.create(F::as_bytes(&a_row));
    let b_h = client.create(F::as_bytes(&b));
    let out_h = client.empty(elems * core::mem::size_of::<F>());

    unsafe {
        let dim = CubeDim::new(std::cmp::max(((elems as u32) + vec as u32 - 1) / vec as u32, 1), 1, 1);
        add_tensor_2d::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(1,1,1),
            dim,
            TensorArg::from_raw_parts::<F>(&a_h, &strides_a, &shape_a, vec),
            TensorArg::from_raw_parts::<F>(&b_h, &strides_out, &shape_out, vec),
            TensorArg::from_raw_parts::<F>(&out_h, &strides_out, &shape_out, vec),
        );
    }
    future::block_on(client.sync());

    let out_b = client.read_one(out_h);
    let out = F::from_bytes(&out_b);
    // Expected CPU result with broadcast along M (rows)
    let mut expect: Vec<F> = Vec::with_capacity(elems);
    for i in 0..m { for j in 0..n { expect.push(F::new((j as f32) + ((i as f32) * 2.0 + j as f32))); } }
    let approx = |u:&[F], v:&[F]| -> bool {
        if u.len()!=v.len() { return false; }
        u.iter().zip(v.iter()).all(|(x,y)| {
            let ux: f32 = (*x).to_f32();
            let vy: f32 = (*y).to_f32();
            (ux - vy).abs() < 1e-3 * f32::max(1.0, vy.abs())
        })
    };
    println!("[rank2 f32 vec={}] add(1xN + MxN) ok? {}", vec, approx(out, &expect));
}
