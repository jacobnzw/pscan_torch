from pathlib import Path

import torch
from max.torch import CustomOpLibrary

# Load our custom operations
mojo_kernels = Path("op")
ops = CustomOpLibrary(mojo_kernels)


def prefix_sum_multiblock_pytorch(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for prefix_sum_multiblock mojo kernel.

    This function implements a multi-block parallel prefix sum (scan) operation.
    The output tensor is larger than the input to accommodate block sums needed
    for the multi-block algorithm.

    Args:
        input_tensor (torch.Tensor): Input tensor to compute prefix sum on.
            Must be a 1D tensor on CUDA device.

    Returns:
        torch.Tensor: Output tensor containing the prefix sum results.
            The size will be input_size + num_blocks due to block sums storage.
    """
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


def prefix_sum_simple_pytorch(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for prefix_sum_simple mojo kernel.

    This function implements a simple single-block parallel prefix sum (scan) operation.
    Limited to input sizes that can fit in a single thread block's shared memory.

    Args:
        input_tensor (torch.Tensor): Input tensor to compute prefix sum on.
            Must be a 1D tensor on CUDA device with size <= thread block size.

    Returns:
        torch.Tensor: Output tensor containing the prefix sum results.
            Same size as input tensor.
    """
    # Create output tensor with same shape as input
    output_tensor = torch.empty_like(input_tensor)

    # Call our custom conv1d operation with explicit output tensor
    # The Mojo signature expects: (out, input)
    # "prefix_sum_simple" matches the @compiler.register("prefix_sum_simple") in op/prefix_sum_simple.mojo
    prefix_sum_simple = ops.prefix_sum_simple[{"size": input_tensor.shape[0]}]
    torch.compile(prefix_sum_simple)(output_tensor, input_tensor)

    return output_tensor


# TODO: Maybe compare against https://github.com/glassroom/torch_parallel_scan ?
