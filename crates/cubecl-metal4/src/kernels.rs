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
  uint2 tgid [[thread_position_in_grid]]) {
    constexpr auto matmulDescriptor = tensor_ops::matmul2d_descriptor(64, 32, 0);
    tensor_ops::matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
    auto mA = a.slice(0, static_cast<int>(tgid.y) * 64);
    auto mB = b.slice(static_cast<int>(tgid.x) * 32, 0);
    auto mC = c.slice(static_cast<int>(tgid.x) * 32, static_cast<int>(tgid.y) * 64);
    matmulOp.run(mA, mB, mC);
}
"#;
    src.to_string()
}

