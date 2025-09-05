from pathlib import Path

import numpy as np
from max.torch import CustomOpLibrary

import torch

# TODO: one file for wrappers one for tests


def prefix_sum_multiblock_pytorch(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for prefix_sum_multiblock mojo kernel.
    """
    # Load our custom operations
    mojo_kernels = Path(__file__).parent.parent / "op"
    ops = CustomOpLibrary(mojo_kernels)

    # Create output tensor with same shape as input
    # TODO: won't we be missing the necessary space for the block sums?
    output_tensor = torch.empty_like(input_tensor)

    # Call our custom conv1d operation with explicit output tensor
    # The Mojo signature expects: (out, input)
    # "prefix_sum_multiblock" matches the @compiler.register("prefix_sum_multiblock") in op/prefix_sum_multiblock.mojo
    prefix_sum_multiblock = ops.prefix_sum_multiblock
    torch.compile(prefix_sum_multiblock)(output_tensor, input_tensor)

    return output_tensor


if __name__ == "__main__":
    INPUT_SIZE = 15  # TODO: Generalize to any size

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
        pytorch_result = prefix_sum_multiblock_pytorch(input_tensor)
        pytorch_result_cpu = pytorch_result.cpu().numpy()
        print(f"PyTorch custom op result: {pytorch_result_cpu}")

        # TODO: Verify PyTorch result
        # np.testing.assert_allclose(pytorch_result_cpu, numpy_result, rtol=1e-5)
        # print("✅ PyTorch custom op verification PASSED")

    except Exception as e:
        print(f"❌ PyTorch custom op failed: {e}")
        pytorch_result_cpu = None
