use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(thiserror::Error, Debug)]
pub enum SoftmaxSetupError {
    #[error(
        "input and output must be 2D tensors with identical shape (got in={in_shape:?}, out={out_shape:?})"
    )]
    InvalidShapes {
        in_shape: Vec<usize>,
        out_shape: Vec<usize>,
    },
}

#[cube(launch_unchecked)]
fn softmax_rows_kernel<E: Float>(input: &Tensor<Line<E>>, output: &mut Tensor<E>, cols: u32) {
    let rank = input.rank();
    let rows = input.shape(rank - 2);
    let row = CUBE_POS_Y;
    if row >= rows {
        terminate!();
    }
    let tid = UNIT_POS_X;
    let stride_in = input.stride(rank - 2);
    let stride_out = output.stride(rank - 2);
    let mut mx = E::from_int(0);
    let mut has = false.runtime();
    let mut j = tid;
    while j < cols {
        let idx = row * stride_in + j;
        let v = input[idx][0];
        if has {
            if v > mx {
                mx = v;
            }
        } else {
            mx = v;
            has = true;
        }
        j += CUBE_DIM_X;
    }
    let mx_all = plane_max(mx);
    let mut denom = E::from_int(0);
    j = tid;
    while j < cols {
        let idx = row * stride_in + j;
        let x = input[idx][0];
        denom += E::exp(x - mx_all);
        j += CUBE_DIM_X;
    }
    let denom_all = plane_sum(denom);
    j = tid;
    while j < cols {
        let in_idx = row * stride_in + j;
        let out_idx = row * stride_out + j;
        let x = input[in_idx][0];
        output[out_idx] = E::exp(x - mx_all) / denom_all;
        j += CUBE_DIM_X;
    }
}

pub fn launch_rows_ref<R: Runtime, E: Float + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
) -> Result<(), SoftmaxSetupError> {
    if input.shape != output.shape || input.shape.len() != 2 {
        return Err(SoftmaxSetupError::InvalidShapes {
            in_shape: input.shape.to_vec(),
            out_shape: output.shape.to_vec(),
        });
    }
    let cols = input.shape[1] as u32;
    let count = CubeCount::Static(1, input.shape[0] as u32, 1);
    let dim = CubeDim::new(256, 1, 1);
    unsafe {
        softmax_rows_kernel::launch_unchecked::<E, R>(
            client,
            count,
            dim,
            input.as_tensor_arg(1),
            output.as_tensor_arg(1),
            cubecl_core::prelude::ScalarArg { elem: cols },
        );
    }
    Ok(())
}
