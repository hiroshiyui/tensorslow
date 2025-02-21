#!/usr/bin/env python
import tensorflow as tf
import time

def matrix_multiplication_benchmark(matrix_size=2048, iterations=10):  # Increased matrix size for more compute
    """Benchmarks matrix multiplication with TensorFlow."""

    # Create two large matrices
    a = tf.random.normal(shape=(matrix_size, matrix_size))
    b = tf.random.normal(shape=(matrix_size, matrix_size))

    # Force eager execution (for consistent timing)
    tf.config.run_functions_eagerly(True)  # Important for accurate timing


    start_time = time.time()
    for _ in range(iterations):
        c = tf.matmul(a, b)  # The matrix multiplication
        # We need to force execution so that the time is measured correctly
        c.numpy() # or tf.identity(c) # or tf.print(tf.reduce_sum(c))

    end_time = time.time()
    elapsed_time = (end_time - start_time) / iterations

    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Average time per multiplication: {elapsed_time:.4f} seconds")

    tf.config.run_functions_eagerly(False) # Restore default behavior

    return elapsed_time


# Run the benchmark
matrix_multiplication_benchmark()  # You can adjust matrix_size and iterations

# Compare to a smaller size to see the scaling
matrix_multiplication_benchmark(matrix_size=1024)

# And even smaller
matrix_multiplication_benchmark(matrix_size=512)

# If you have multiple CPUs, you can try pinning to a specific one:
# with tf.device('/physical_device:CPU:0'): # Example for CPU 0
#     matrix_multiplication_benchmark()
