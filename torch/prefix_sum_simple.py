from pathlib import Path

import numpy as np
from max.torch import CustomOpLibrary

import torch


def prefix_sum_simple_pytorch(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for prefix_sum_simple mojo kernel.
    """
    # Load our custom operations
    mojo_kernels = Path(__file__).parent.parent / "op"
    ops = CustomOpLibrary(mojo_kernels)

    # Create output tensor with same shape as input
    output_tensor = torch.empty_like(input_tensor)

    # Call our custom conv1d operation with explicit output tensor
    # The Mojo signature expects: (out, input)
    # "prefix_sum_simple" matches the @compiler.register("prefix_sum_simple") in op/prefix_sum_simple.mojo
    prefix_sum_simple = ops.prefix_sum_simple[{"size": input_tensor.shape[0]}]
    torch.compile(prefix_sum_simple)(output_tensor, input_tensor)

    return output_tensor


# TODO: Maybe compare against https://github.com/glassroom/torch_parallel_scan ?
# def compute_numpy_reference(
#     input_array: np.ndarray, kernel_array: np.ndarray
# ) -> np.ndarray:
#     """NumPy reference implementation for verification."""
#     INPUT_SIZE = len(input_array)
#     KERNEL_SIZE = len(kernel_array)

#     expected_result = np.zeros_like(input_array, dtype=np.float32)
#     for i in range(INPUT_SIZE):
#         for j in range(KERNEL_SIZE):
#             if i + j < INPUT_SIZE:
#                 expected_result[i] += input_array[i + j] * kernel_array[j]
#     return expected_result


if __name__ == "__main__":
    INPUT_SIZE = 15  # TODO: Current Mojo kernel is using TPB=8 and only 1 block.

    input_array = np.arange(INPUT_SIZE, dtype=np.float32)

    print(f"Input array: {input_array}")
    print()

    # numpy_result = compute_numpy_reference(input_array)
    # print(f"NumPy reference result: {numpy_result}")
    # print()

    device = "cuda"
    input_tensor = torch.from_numpy(input_array).to(device)

    print(f"Testing PyTorch Custom Op (device: {device})")
    print("-" * 40)

    try:
        pytorch_result = prefix_sum_simple_pytorch(input_tensor)
        pytorch_result_cpu = pytorch_result.cpu().numpy()
        print(f"PyTorch custom op result: {pytorch_result_cpu}")

        # TODO: Verify PyTorch result
        # np.testing.assert_allclose(pytorch_result_cpu, numpy_result, rtol=1e-5)
        # print("✅ PyTorch custom op verification PASSED")

    except Exception as e:
        print(f"❌ PyTorch custom op failed: {e}")
        pytorch_result_cpu = None
