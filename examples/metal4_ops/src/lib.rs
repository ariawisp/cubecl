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

// Additional unary kernels (float)
#[cube(launch_unchecked)]
fn abs_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Abs::abs(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn exp_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Exp::exp(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn log_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Log::log(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn log1p_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Log1p::log1p(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn recip_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Recip::recip(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn tanh_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Tanh::tanh(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn sqrt_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Sqrt::sqrt(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn floor_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Floor::floor(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn ceil_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Ceil::ceil(a[ABSOLUTE_POS]); }
}

#[cube(launch_unchecked)]
fn round_array<F: Float>(a: &Array<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = Round::round(a[ABSOLUTE_POS]); }
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

        // Extra unary outputs
        let out_abs = client.empty(n * core::mem::size_of::<F>());
        let out_exp = client.empty(n * core::mem::size_of::<F>());
        let out_log = client.empty(n * core::mem::size_of::<F>());
        let out_log1p = client.empty(n * core::mem::size_of::<F>());
        let out_recip = client.empty(n * core::mem::size_of::<F>());
        let out_tanh = client.empty(n * core::mem::size_of::<F>());
        let out_sqrt = client.empty(n * core::mem::size_of::<F>());
        let out_floor = client.empty(n * core::mem::size_of::<F>());
        let out_ceil = client.empty(n * core::mem::size_of::<F>());
        let out_round = client.empty(n * core::mem::size_of::<F>());

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

            // Unary suite
            abs_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_abs, n, vec));
            exp_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_exp, n, vec));
            log_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_log, n, vec));
            log1p_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_log1p, n, vec));
            recip_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_recip, n, vec));
            tanh_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_tanh, n, vec));
            sqrt_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_sqrt, n, vec));
            floor_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_floor, n, vec));
            ceil_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_ceil, n, vec));
            round_array::launch_unchecked::<F, R>(&client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<F>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<F>(&out_round, n, vec));
        }

        future::block_on(client.sync());

        let add_b = client.read_one(out_add);
        let sub_b = client.read_one(out_sub);
        let mul_b = client.read_one(out_mul);
        let div_b = client.read_one(out_div);
        let neg_b = client.read_one(out_neg);
        let add_bcast_b = client.read_one(out_add_bcast);
        let abs_b = client.read_one(out_abs);
        let exp_b = client.read_one(out_exp);
        let log_b = client.read_one(out_log);
        let log1p_b = client.read_one(out_log1p);
        let recip_b = client.read_one(out_recip);
        let tanh_b = client.read_one(out_tanh);
        let sqrt_b = client.read_one(out_sqrt);
        let floor_b = client.read_one(out_floor);
        let ceil_b = client.read_one(out_ceil);
        let round_b = client.read_one(out_round);
        let add = F::from_bytes(&add_b);
        let sub = F::from_bytes(&sub_b);
        let mul = F::from_bytes(&mul_b);
        let div = F::from_bytes(&div_b);
        let neg = F::from_bytes(&neg_b);
        let add_bcast = F::from_bytes(&add_bcast_b);
        let abs_v = F::from_bytes(&abs_b);
        let exp_v = F::from_bytes(&exp_b);
        let log_v = F::from_bytes(&log_b);
        let log1p_v = F::from_bytes(&log1p_b);
        let recip_v = F::from_bytes(&recip_b);
        let tanh_v = F::from_bytes(&tanh_b);
        let sqrt_v = F::from_bytes(&sqrt_b);
        let floor_v = F::from_bytes(&floor_b);
        let ceil_v = F::from_bytes(&ceil_b);
        let round_v = F::from_bytes(&round_b);

        let exp_add: Vec<F> = a.iter().zip(b.iter()).map(|(x,y)| *x + *y).collect();
        let exp_sub: Vec<F> = a.iter().zip(b.iter()).map(|(x,y)| *x - *y).collect();
        let exp_mul: Vec<F> = a.iter().zip(b.iter()).map(|(x,y)| *x * *y).collect();
        let exp_div: Vec<F> = a.iter().zip(b.iter()).map(|(x,y)| *x / *y).collect();
        let exp_neg: Vec<F> = a.iter().map(|x| -*x).collect();
        let exp_add_bcast: Vec<F> = a.iter().map(|x| *x + b_broadcast[0]).collect();

        let approx_eq = |u:&[F], v:&[F]| -> bool {
            if u.len() != v.len() { return false; }
            u.iter().zip(v.iter()).all(|(x,y)| {
                let ux: f32 = ToF32::to_f32(*x);
                let vy: f32 = ToF32::to_f32(*y);
                (ux - vy).abs() < 1e-3 * f32::max(1.0, vy.abs())
            })
        };
        println!("[float {:?} vec={}] add={} sub={} mul={} div={} neg={} add(bcast)={}",
                 core::any::type_name::<F>(), vec,
                 approx_eq(&add, &exp_add), approx_eq(&sub, &exp_sub),
                 approx_eq(&mul, &exp_mul), approx_eq(&div, &exp_div),
                 approx_eq(&neg, &exp_neg), approx_eq(&add_bcast, &exp_add_bcast));

        // Expected values for unary suite
        let exp_abs: Vec<F> = a.iter().map(|x| F::new(ToF32::to_f32(*x).abs())).collect();
        let exp_exp: Vec<F> = a.iter().map(|x| F::new(ToF32::to_f32(*x).exp())).collect();
        let exp_log: Vec<F> = a.iter().map(|x| F::new(ToF32::to_f32(*x).ln())).collect();
        let exp_log1p: Vec<F> = a.iter().map(|x| F::new((1.0 + ToF32::to_f32(*x)).ln())).collect();
        let exp_recip: Vec<F> = a.iter().map(|x| F::new(1.0 / ToF32::to_f32(*x))).collect();
        let exp_tanh: Vec<F> = a.iter().map(|x| F::new(ToF32::to_f32(*x).tanh())).collect();
        let exp_sqrt: Vec<F> = a.iter().map(|x| F::new(ToF32::to_f32(*x).sqrt())).collect();
        let exp_floor: Vec<F> = a.iter().map(|x| F::new(ToF32::to_f32(*x).floor())).collect();
        let exp_ceil: Vec<F> = a.iter().map(|x| F::new(ToF32::to_f32(*x).ceil())).collect();
        let exp_round: Vec<F> = a.iter().map(|x| F::new(ToF32::to_f32(*x).round())).collect();
        println!(
            "[float {:?} vec={}] abs={} exp={} log={} log1p={} recip={} tanh={} sqrt={} floor={} ceil={} round={}",
            core::any::type_name::<F>(), vec,
            approx_eq(&abs_v, &exp_abs), approx_eq(&exp_v, &exp_exp), approx_eq(&log_v, &exp_log), approx_eq(&log1p_v, &exp_log1p),
            approx_eq(&recip_v, &exp_recip), approx_eq(&tanh_v, &exp_tanh), approx_eq(&sqrt_v, &exp_sqrt),
            approx_eq(&floor_v, &exp_floor), approx_eq(&ceil_v, &exp_ceil), approx_eq(&round_v, &exp_round)
        );
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

    // Integer i32 edge cases (negative / positive division)
    fn run_int_i32_edges<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>, vec: u8) {
        let a: Vec<i32> = (-9..= -1).collect();
        let b: Vec<i32> = (2..=10).collect();
        let n = a.len();
        let out_div = client.empty(n * core::mem::size_of::<i32>());
        let a_handle = client.create(i32::as_bytes(&a));
        let b_handle = client.create(i32::as_bytes(&b));
        #[cube(launch_unchecked)]
        fn div_array_i32(a: &Array<Line<i32>>, b: &Array<Line<i32>>, out: &mut Array<Line<i32>>) {
            if ABSOLUTE_POS < a.len() { out[ABSOLUTE_POS] = a[ABSOLUTE_POS] / b[ABSOLUTE_POS]; }
        }
        unsafe {
            let dim = CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1);
            div_array_i32::launch_unchecked::<R>(client, CubeCount::Static(1,1,1), dim,
                ArrayArg::from_raw_parts::<i32>(&a_handle, n, vec),
                ArrayArg::from_raw_parts::<i32>(&b_handle, n, vec),
                ArrayArg::from_raw_parts::<i32>(&out_div, n, vec));
        }
        future::block_on(client.sync());
        let div_b = client.read_one(out_div);
        let div = i32::from_bytes(&div_b);
        let exp_div: Vec<i32> = a.iter().zip(b.iter()).map(|(x,y)| x / y).collect();
        println!("[int i32 vec={}] neg/pos div={}", vec, div == exp_div);
    }

    // Sweep vectorization and types
    for &vec in &[1u8, 2, 4, 8] {
        run_float_one::<R, f32>(&client, vec);
        run_int_one::<R, u8>(&client, vec);
        run_int_one::<R, i32>(&client, vec);
        run_int_i32_edges::<R>(&client, vec);
        #[cfg(feature = "metal4")]
        run_float_one::<R, f16>(&client, vec);
    }

    // Rank-2 broadcast demos across vec factors
    for &vec in &[1u8, 4, 8] {
        // A shape (M,N), B shape (M,1)
        run_rank2_broadcast::<R, f32>(&client, 3, 4, vec);
        // A shape (1,N), B shape (M,N)
        run_rank2_broadcast_row::<R, f32>(&client, 3, 5, vec);
    }

    #[cfg(feature = "metal4")]
    {
        run_mpp_matmul::<R>(&client);
    }

    // (Cast kernels deferred) – keeping runtime focused tests runnable.

    // Rank-1 strided copy (Tensor path)
    for &vec in &[1u8, 4] {
        run_rank1_strided::<R, f32>(&client, 16, 2, vec);
    }

    // Reduction 1D demo: sum to scalar
    run_reduce_sum_1d::<R, f32>(&client, 37);
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
            let ux: f32 = ToF32::to_f32(*x);
            let vy: f32 = ToF32::to_f32(*y);
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
            let ux: f32 = ToF32::to_f32(*x);
            let vy: f32 = ToF32::to_f32(*y);
            (ux - vy).abs() < 1e-3 * f32::max(1.0, vy.abs())
        })
    };
    println!("[rank2 f32 vec={}] add(1xN + MxN) ok? {}", vec, approx(out, &expect));
}


// Rank-1 strided copy kernel (Tensor path)
#[cube(launch_unchecked)]
fn copy_tensor_1d<F: Float>(src: &Tensor<Line<F>>, dst: &mut Tensor<Line<F>>) {
    if ABSOLUTE_POS < dst.len() { dst[ABSOLUTE_POS] = src[ABSOLUTE_POS]; }
}

fn run_rank1_strided<R: Runtime, F: Float + CubeElement + ToF32>(client: &ComputeClient<R::Server, R::Channel>, n: usize, stride: usize, vec: u8) {
    // Underlying buffer size must accommodate stride
    let buf_len = (n - 1) * stride + 1;
    let mut src_full: Vec<F> = Vec::with_capacity(buf_len);
    for i in 0..buf_len { src_full.push(F::new(i as f32)); }
    // Strided view over src_full: represent as rank-2 (n,1) so rank-2 path uses strides
    let shape = vec![n, 1];
    let strides = vec![stride, 1];
    let src_h = client.create(F::as_bytes(&src_full));
    let dst_h = client.empty(n * core::mem::size_of::<F>());
    unsafe {
        let dim = CubeDim::new(std::cmp::max((n as u32 + vec as u32 - 1) / vec as u32, 1), 1, 1);
        copy_tensor_1d::launch_unchecked::<F, R>(client, CubeCount::Static(1,1,1), dim,
            TensorArg::from_raw_parts::<F>(&src_h, &strides, &shape, vec),
            TensorArg::from_raw_parts::<F>(&dst_h, &vec![1usize, 1usize], &shape, vec));
    }
    future::block_on(client.sync());
    let out_b2 = client.read_one(dst_h);
    let out = F::from_bytes(&out_b2);
    let mut exp: Vec<F> = Vec::with_capacity(n);
    for i in 0..n { exp.push(src_full[i * stride]); }
    let approx = |u:&[F], v:&[F]| u.len()==v.len() && u.iter().zip(v.iter()).all(|(x,y)| (ToF32::to_f32(*x) - ToF32::to_f32(*y)).abs() < 1e-6);
    println!("[rank1 stride vec={}] copy 1D stride={} ok? {}", vec, stride, approx(&out, &exp));
}

// Minimal 1D reduction demo (special-cased by MSL4 codegen)
#[cube(launch_unchecked)]
fn reduce_sum_1d<F: Float>(a: &Tensor<Line<F>>, out: &mut Array<Line<F>>) {
    if ABSOLUTE_POS < out.len() { out[ABSOLUTE_POS] = F::new(0.0); }
}

fn run_reduce_sum_1d<R: Runtime, F: Float + CubeElement + ToF32>(client: &ComputeClient<R::Server, R::Channel>, n: usize) {
    let a: Vec<F> = (0..n).map(|i| F::new((i as f32) * 0.25 + 0.5)).collect();
    let a_h = client.create(F::as_bytes(&a));
    let out_h = client.empty(1 * core::mem::size_of::<F>());
    let shape = vec![n];
    let strides = vec![1usize];
    unsafe {
        let dim = CubeDim::new(1, 1, 1);
        reduce_sum_1d::launch_unchecked::<F, R>(client, CubeCount::Static(1,1,1), dim,
            TensorArg::from_raw_parts::<F>(&a_h, &strides, &shape, 1),
            ArrayArg::from_raw_parts::<F>(&out_h, 1, 1));
    }
    future::block_on(client.sync());
    let out_b = client.read_one(out_h);
    let out = F::from_bytes(&out_b);
    let got = out[0].to_f32();
    let exp = a.iter().map(|x| x.to_f32()).sum::<f32>();
    let ok = (got - exp).abs() < 1e-4 * f32::max(1.0, exp.abs());
    println!("[reduce sum 1d] n={} ok? {}", n, ok);
}

#[cfg(feature = "metal4")]
#[cube(launch_unchecked)]
fn mpp_matmul_driver(a: &Tensor<Line<half::f16>>, b: &Tensor<Line<half::f16>>, out: &mut Tensor<Line<f32>>) {
    // Dummy body; server swaps to MPP kernel when types and metadata match.
    if ABSOLUTE_POS < out.len() { out[ABSOLUTE_POS] = out[ABSOLUTE_POS]; }
}

#[cfg(feature = "metal4")]
fn run_mpp_matmul<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) {
    use half::f16;
    // Dimensions: M x K  times  K x N  ->  M x N
    let m = 64usize;
    let n = 32usize;
    let k = 33usize; // dynamic K
    let shape_a = vec![m, k];
    let shape_b = vec![k, n];
    let shape_c = vec![m, n];
    let strides_a = compact_strides(&shape_a);
    let strides_b = compact_strides(&shape_b);
    let strides_c = compact_strides(&shape_c);

    // Initialize inputs
    let mut a: Vec<f16> = Vec::with_capacity(m * k);
    let mut b: Vec<f16> = Vec::with_capacity(k * n);
    for i in 0..m { for kk in 0..k { a.push(f16::from_f32(((i + kk) % 7) as f32 * 0.25)); } }
    for kk in 0..k { for j in 0..n { b.push(f16::from_f32(((kk + j) % 5) as f32 * 0.5)); } }
    // CPU reference (f32)
    let mut c_ref: Vec<f32> = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let va = f32::from(a[i * k + kk]);
                let vb = f32::from(b[kk * n + j]);
                acc += va * vb;
            }
            c_ref[i * n + j] = acc;
        }
    }
    let a_h = client.create(f16::as_bytes(&a));
    let b_h = client.create(f16::as_bytes(&b));
    let c_h = client.empty(shape_c.iter().product::<usize>() * core::mem::size_of::<f32>());

    unsafe {
        // vec=1 to keep simple; server infers MPP grid from metadata
        let dim = CubeDim::new(1, 1, 1);
        mpp_matmul_driver::launch_unchecked::<R>(
            client,
            CubeCount::Static(1,1,1),
            dim,
            TensorArg::from_raw_parts::<f16>(&a_h, &strides_a, &shape_a, 1),
            TensorArg::from_raw_parts::<f16>(&b_h, &strides_b, &shape_b, 1),
            TensorArg::from_raw_parts::<f32>(&c_h, &strides_c, &shape_c, 1),
        );
    }
    future::block_on(client.sync());
    let c_b = client.read_one(c_h);
    let c = f32::from_bytes(&c_b);
    let approx = |u:&[f32], v:&[f32]| -> bool {
        if u.len()!=v.len() { return false; }
        u.iter().zip(v.iter()).all(|(x,y)| (x - y).abs() < 1e-2 * f32::max(1.0, y.abs()))
    };
    println!("[mpp matmul] MxK={}x{} * KxN={}x{} ok? {}", m, k, k, n, approx(&c, &c_ref));
}
