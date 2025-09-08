from pathlib import Path

import numpy as np
from max.torch import CustomOpLibrary

import torch

# TODO: one file for wrappers one for tests


# def get_launch_config(input_size: int) -> tuple[int, int]:
#     props = torch.cuda.get_device_properties()
#     warp_size = props.warp_size
#     MAX_TPB = 1024
#     optimal_tpb = min(MAX_TPB, ((input_size + warp_size - 1) // warp_size) * warp_size)
#     optimal_tpb = 8
#     blocks_needed = (input_size + optimal_tpb - 1) // optimal_tpb
#     return blocks_needed, optimal_tpb


def prefix_sum_multiblock_pytorch(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for prefix_sum_multiblock mojo kernel.
    """
    # Load our custom operations
    mojo_kernels = Path(__file__).parent.parent / "op"
    ops = CustomOpLibrary(mojo_kernels)

    # Output tensor needs space for block sums
    tpb = 8
    blocks_needed = (input_tensor.shape[0] + tpb - 1) // tpb
    output_shape = list(input_tensor.shape)
    output_shape[0] += blocks_needed
    print(f"output_shape: {output_shape}")
    # new_empty copies the dtype and device from the input tensor
    # both input and output tensors are created on the GPU
    output_tensor = input_tensor.new_empty(output_shape)

    # Call our custom prefix_sum_multiblock operation with explicit output tensor
    # The Mojo signature expects: (out, input)
    # "prefix_sum_multiblock" matches the @compiler.register("prefix_sum_multiblock") in op/prefix_sum_multiblock.mojo
    prefix_sum_multiblock = ops.prefix_sum_multiblock
    torch.compile(prefix_sum_multiblock)(output_tensor, input_tensor)

    return output_tensor


if __name__ == "__main__":
    INPUT_SIZE = 30  # TODO: Generalize to any size

    input_array = np.arange(INPUT_SIZE, dtype=np.float32)

    print(f"Input array: {input_array}")
    print()

    numpy_result = input_array.cumsum()
    print(f"NumPy reference result: {numpy_result}")
    print()

    # input tensors created on the GPU
    device = "cuda"
    input_tensor = torch.from_numpy(input_array).to(device)

    print(f"Testing PyTorch Custom Op (device: {device})")
    print("-" * 40)

    try:
        pytorch_result = prefix_sum_multiblock_pytorch(input_tensor)
        pytorch_result_cpu = pytorch_result.cpu().numpy()
        print(f"PyTorch custom op result: {pytorch_result_cpu}")

        # TODO: Verify PyTorch result
        # np.testing.assert_allclose(pytorch_result_cpu, numpy_result, rtol=1e-5)
        # print("✅ PyTorch custom op verification PASSED")

    except Exception as e:
        print(f"❌ PyTorch custom op failed: {e}")
        pytorch_result_cpu = None
