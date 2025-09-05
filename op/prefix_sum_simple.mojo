from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import size_of, argv
from math import log2

alias TPB = 8
alias SIZE = 8
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


fn prefix_sum_simple[
    layout: Layout
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    size: Int,  # used for bounds checking
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    shared = tb[dtype]().row_major[TPB]().shared().alloc()
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    offset = 1
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val: output.element_type = 0
        if local_i >= offset and local_i < size:
            current_val = shared[local_i - offset]  # read

        barrier()
        if local_i >= offset and local_i < size:
            shared[local_i] += current_val

        barrier()
        offset *= 2

    if global_i < size:
        output[global_i] = shared[local_i]


import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer

alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)


@compiler.register("prefix_sum_simple")
struct PrefixSumSimpleOp:
    @staticmethod
    fn execute[
        target: StaticString,  # The kind of device this will be run on: "cpu" or "gpu"
        size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[
            dtype=dtype, rank=1
        ],  # NOTE: rank=1 limits to vectors/lists
        input: InputTensor[dtype=dtype, rank = output.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        out_tensor = output.to_layout_tensor()
        input_tensor = input.to_layout_tensor()
        alias layout = input_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output.dtype]]](out_tensor.ptr),
                    size,
                    owning=False,
                ),
                0,
            )
            # launching the kernel
            gpu_ctx.enqueue_function[prefix_sum_simple[layout]](
                out_tensor,
                input_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )

        elif target == "cpu":
            # we can fallback to CPU
            print("pscan: falling back to CPU")
            pass
        else:
            raise Error("Unsupported target: " + target)
