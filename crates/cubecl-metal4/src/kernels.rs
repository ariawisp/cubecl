pub fn mpp_matmul_2d_source() -> String {
    let src = r#"#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

[[ kernel ]] void mpp_matmul_2d(
  tensor<device half, dextents<int, 2>> a [[ buffer(0) ]],
  tensor<device half, dextents<int, 2>> b [[ buffer(1) ]],
  tensor<device float, dextents<int, 2>> c [[ buffer(2) ]],
  uint2 tgpos [[threadgroup_position_in_grid]]) {
    constexpr auto matmulDescriptor = tensor_ops::matmul2d_descriptor(64, 32, dynamic_length_v<int>);
    tensor_ops::matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
    auto mA = a.slice(0, static_cast<int>(tgpos.y) * 64);
    auto mB = b.slice(static_cast<int>(tgpos.x) * 32, 0);
    auto mC = c.slice(static_cast<int>(tgpos.x) * 32, static_cast<int>(tgpos.y) * 64);
    matmulOp.run(mA, mB, mC);
}
"#;

    src.to_string()
}

pub fn mpp_convolution2d_source(
    n: i32,
    h: i32,
    w: i32,
    c: i32,
    kh: i32,
    kw: i32,
    o: i32,
    hout: i32,
    wout: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
    groups: i32,
) -> String {
    let src = format!(
        r#"#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

// MPP convolution2d (NHWC activation, HWIO weights, NHWO destination)
[[ kernel ]] void mpp_convolution2d(
  tensor<device half, dextents<int, 4>> activation [[ buffer(0) ]],
  tensor<device half, dextents<int, 4>> weights [[ buffer(1) ]],
  tensor<device float, dextents<int, 4>> destination [[ buffer(2) ]],
  uint3 tgid [[thread_position_in_grid]]) {{
    constexpr auto desc = tensor_ops::convolution2d_descriptor(
        int4({hout}, {wout}, {o}, {n}),
        int4({h}, {w}, {c}, {n}),
        int2({kh}, {kw}),
        tensor_ops::convolution2d_activation_layout::nhwc,
        tensor_ops::convolution2d_weights_layout::hwio,
        int2({stride_h}, {stride_w}),
        int2({dilation_h}, {dilation_w}),
        {groups},
        false,
        tensor_ops::convolution2d_descriptor::mode::multiply);
    tensor_ops::convolution2d<desc, execution_simdgroups<4>> convOp;
    convOp.run(activation, weights, destination);
}}
"#,
        n = n,
        h = h,
        w = w,
        c = c,
        kh = kh,
        kw = kw,
        o = o,
        hout = hout,
        wout = wout,
        stride_h = stride_h,
        stride_w = stride_w,
        dilation_h = dilation_h,
        dilation_w = dilation_w,
        groups = groups,
    );
    src
}
