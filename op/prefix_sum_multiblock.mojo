from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import size_of, argv
from math import log2
from testing import assert_equal


# Hillis Steel parallel scan with sum as associative operator.


alias SIZE = 30
alias TPB = 8
alias BLOCKS = 4
alias BLOCKS_PER_GRID = (BLOCKS, 1)
alias dtype = DType.float32


# Kernel 1: Compute local prefix sums and store block sums in out
fn prefix_sum_local_phase[
    out_layout: Layout, in_layout: Layout
](
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # TPB has to be a compile-time constant for shared memory allocation
    shared = tb[dtype]().row_major[TPB]().shared().alloc()

    # Load data into shared memory
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    # Compute local prefix sum using parallel reduction
    offset = 1
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val: output.element_type = 0
        if local_i >= offset and local_i < TPB:
            current_val = shared[local_i - offset]  # read

        barrier()
        if local_i >= offset and local_i < TPB:
            shared[local_i] += current_val  # write

        barrier()
        offset *= 2

    # Write local results to output
    if global_i < size:
        output[global_i] = shared[local_i]

    # Store block sums in auxiliary space
    if local_i == TPB - 1:
        output[size + block_idx.x] = shared[local_i]


# Kernel 2: Add block sums to their respective blocks
fn prefix_sum_block_sum_phase[
    layout: Layout
](output: LayoutTensor[mut=False, dtype, layout], size: Int):
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # Second pass: add previous block's sum to each element
    if block_idx.x > 0 and global_i < size:
        # FIX: need to add all all previous block sums
        var prev_block_sum: output.element_type = 0
        for i in range(block_idx.x):
            # NOTE: sum accumulated in prev_block_sum register as it 
            # can't be accumulated directly to output[global_i]
            # Disadvantages: 
            # - Thread divergence (threads with higher block_idx.x do more work)
            # - Threads repeat the same work
            prev_block_sum += output[size + i]
        output[global_i] += prev_block_sum


# ANCHOR_END: prefix_sum_complete_solution

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer, DeviceAttribute


@compiler.register("prefix_sum_multiblock")
struct PrefixSumMultiBlockOp:
    @staticmethod
    fn execute[
        target: StaticString,  # The kind of device this will be run on: "cpu" or "gpu"
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        input: InputTensor[dtype=dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        out_tensor = output.to_layout_tensor()
        input_tensor = input.to_layout_tensor()
        alias out_layout = out_tensor.layout
        alias in_layout = input_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()

            # Compute optimal block size (multiple of warp size)
            # TODO: Inputs size < WARP_SIZE, wastes threads if tbp=WARP_SIZE, no? Matters even?
            size = in_layout.size()
            # WARP_SIZE = gpu_ctx.get_attribute(DeviceAttribute.WARP_SIZE)
            # MAX_TPB = gpu_ctx.get_attribute(
            #     DeviceAttribute.MAX_THREADS_PER_BLOCK
            # )
            # optimal_tpb = min(
            #     MAX_TPB,
            #     ((size + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE,
            # )
            # blocks_needed = (size + optimal_tpb - 1) // optimal_tpb
            # print("size:", size)
            # print("max_tpb:", MAX_TPB)
            optimal_tpb = TPB
            blocks_needed = (size + optimal_tpb - 1) // optimal_tpb
            print("optimal_tpb:", optimal_tpb)
            print("blocks_needed:", blocks_needed)
            #

            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output.dtype]]](out_tensor.ptr),
                    out_layout.size(),
                    owning=False,
                ),
                0,
            )
            # Phase 1: Local prefix sums
            gpu_ctx.enqueue_function[
                prefix_sum_local_phase[out_layout, in_layout]
            ](
                out_tensor,
                input_tensor,
                size,
                grid_dim=blocks_needed,
                block_dim=optimal_tpb,
            )

            # Phase 2: Add block sums
            gpu_ctx.enqueue_function[prefix_sum_block_sum_phase[out_layout]](
                out_tensor,
                size,
                grid_dim=blocks_needed,
                block_dim=optimal_tpb,
            )

            gpu_ctx.synchronize()
        elif target == "cpu":
            # we can fallback to CPU
            print("pscan: falling back to CPU")
            pass
        else:
            raise Error("Unsupported target: " + target)


# TODO: move to test folder?
def main():
    alias THREADS_PER_BLOCK = (TPB, 1)
    alias EXTENDED_SIZE = SIZE + BLOCKS
    alias layout = Layout.row_major(SIZE)
    alias extended_layout = Layout.row_major(EXTENDED_SIZE)
    with DeviceContext() as ctx:
        size = SIZE
        num_blocks = (size + TPB - 1) // TPB

        if num_blocks > EXTENDED_SIZE - SIZE:
            raise Error("Extended buffer too small for the number of blocks")

        buffer_size = EXTENDED_SIZE
        out = ctx.enqueue_create_buffer[dtype](buffer_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)

        # Initialize input array as [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())

        var out_tensor = LayoutTensor[mut=False, dtype, extended_layout](
            out.unsafe_ptr()
        )

        # Phase 1: Local prefix sums
        ctx.enqueue_function[
            prefix_sum_local_phase[extended_layout, extended_layout]
        ](
            out_tensor,
            a_tensor,
            size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # Phase 2: Add block sums
        ctx.enqueue_function[prefix_sum_block_sum_phase[extended_layout]](
            out_tensor,
            size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # Verify results for both cases
        expected = ctx.enqueue_create_host_buffer[dtype](size).enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            expected[0] = a_host[0]
            for i in range(1, size):
                expected[i] = expected[i - 1] + a_host[i]

            print("in:", a_host)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("exp:", expected)
            # Here we need to use the size of the original array, not the extended one
            size = SIZE
            for i in range(size):
                assert_equal(out_host[i], expected[i])
