import numpy as np
import torch

from wrappers import prefix_sum_multiblock_pytorch, prefix_sum_simple_pytorch


def compute_numpy_reference(input_array):
    """Compute reference prefix sum using NumPy."""
    return input_array.cumsum()


def test_prefix_sum_multiblock():
    """Test the multiblock prefix sum implementation."""
    INPUT_SIZE = 30  # TODO: Generalize to any size

    input_array = np.arange(INPUT_SIZE, dtype=np.float32)

    print(f"Input array: {input_array}")
    print()

    numpy_result = compute_numpy_reference(input_array)
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

        np.testing.assert_allclose(
            pytorch_result_cpu[:INPUT_SIZE], numpy_result, rtol=1e-5
        )
        print("✅ PyTorch custom op verification PASSED")

    except Exception as e:
        print(f"❌ PyTorch custom op failed: {e}")
        pytorch_result_cpu = None


def test_prefix_sum_simple():
    """Test the simple prefix sum implementation."""
    INPUT_SIZE = 8  # TODO: Current Mojo kernel is using TPB=8 and only 1 block.

    input_array = np.arange(INPUT_SIZE, dtype=np.float32)

    print(f"Input array: {input_array}")
    print()

    numpy_result = compute_numpy_reference(input_array)
    print(f"NumPy reference result: {numpy_result}")
    print()

    device = "cuda"
    input_tensor = torch.from_numpy(input_array).to(device)

    print(f"Testing PyTorch Custom Op (device: {device})")
    print("-" * 40)

    try:
        pytorch_result = prefix_sum_simple_pytorch(input_tensor)
        pytorch_result_cpu = pytorch_result.cpu().numpy()
        print(f"PyTorch custom op result: {pytorch_result_cpu}")

        np.testing.assert_allclose(pytorch_result_cpu, numpy_result, rtol=1e-5)
        print("✅ PyTorch custom op verification PASSED")

    except Exception as e:
        print(f"❌ PyTorch custom op failed: {e}")
        pytorch_result_cpu = None


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Prefix Sum Multiblock")
    print("=" * 50)
    test_prefix_sum_multiblock()

    print("\n" + "=" * 50)
    print("Testing Prefix Sum Simple")
    print("=" * 50)
    test_prefix_sum_simple()
