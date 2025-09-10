from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import size_of, argv
from math import log2
from testing import assert_equal
from op.prefix_sum_multiblock import (
    prefix_sum_local_phase,
    prefix_sum_block_sum_phase,
    TPB,
    SIZE,
    BLOCKS,
    BLOCKS_PER_GRID,
    dtype,
)


def test_prefix_sum_multiblock():
    """
    Standalone test for the prefix_sum_multiblock kernel.
    Tests the kernel directly without going through Python wrappers.
    """
    print("Testing prefix_sum_multiblock kernel directly...")
    print("=" * 50)

    alias THREADS_PER_BLOCK = (TPB, 1)
    alias EXTENDED_SIZE = SIZE + BLOCKS
    alias layout = Layout.row_major(SIZE)
    alias extended_layout = Layout.row_major(EXTENDED_SIZE)

    with DeviceContext() as ctx:
        size = SIZE
        num_blocks = (size + TPB - 1) // TPB

        print("Test configuration:")
        print("  Input size:", size)
        print("  Threads per block:", TPB)
        print("  Number of blocks:", num_blocks)
        print("  Extended buffer size:", EXTENDED_SIZE)
        print()

        if num_blocks > EXTENDED_SIZE - SIZE:
            raise Error("Extended buffer too small for the number of blocks")

        buffer_size = EXTENDED_SIZE
        out = ctx.enqueue_create_buffer[dtype](buffer_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)

        # Initialize input array as [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,...]
        print("Initializing input data...")
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())

        var out_tensor = LayoutTensor[mut=False, dtype, extended_layout](
            out.unsafe_ptr()
        )

        print("Executing Phase 1: Local prefix sums...")
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

        print("Executing Phase 2: Add block sums...")
        # Phase 2: Add block sums
        ctx.enqueue_function[prefix_sum_block_sum_phase[extended_layout]](
            out_tensor,
            size,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        print("Computing expected results...")
        # Compute expected results (CPU reference implementation)
        expected = ctx.enqueue_create_host_buffer[dtype](size).enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            expected[0] = a_host[0]
            for i in range(1, size):
                expected[i] = expected[i - 1] + a_host[i]

            print("Input array:", a_host)

        print("Verifying results...")
        with out.map_to_host() as out_host:
            print("Output array:", out_host)
            print("Expected array:", expected)
            print()

            # Verify results - compare only the first SIZE elements
            verification_passed = True
            for i in range(size):
                try:
                    assert_equal(out_host[i], expected[i])
                except:
                    print(
                        "❌ Mismatch at index",
                        i,
                        ": got",
                        out_host[i],
                        ", expected",
                        expected[i],
                    )
                    verification_passed = False

            if verification_passed:
                print(
                    "✅ All tests PASSED! Kernel produces correct prefix sum"
                    " results."
                )
            else:
                print("❌ Some tests FAILED! Check the kernel implementation.")
                raise Error("Test verification failed")


def main():
    """
    Main function to run the standalone test.
    """
    try:
        test_prefix_sum_multiblock()
        print("\n" + "=" * 50)
        print("🎉 Test completed successfully!")
    except e:
        print("\n" + "=" * 50)
        print("💥 Test failed with error:", e)
        raise e
