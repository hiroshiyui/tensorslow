#!/usr/bin/env python
import tensorflow as tf
import time
import numpy as np

# Define the model (example: a simple dense network)
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Example output size
    ])
    return model

# Define the input shape (adjust as needed)
input_shape = (784,)  # Example: for MNIST-like data
model = create_model(input_shape)

# Generate some dummy data (replace with your actual data)
num_samples = 1000000  # Number of samples for the benchmark
input_data = np.random.rand(num_samples, *input_shape)  # * unpacks the shape tuple
output_data = np.random.randint(0, 10, size=(num_samples,)) # Example: classification

# Convert data to TensorFlow tensors (important for accurate timing)
input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
output_tensor = tf.convert_to_tensor(output_data, dtype=tf.int32)

# Define the loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()  # Appropriate for integer labels
optimizer = tf.keras.optimizers.Adam()

# @tf.function decorator:  Crucial for performance! Compiles the function
@tf.function  # Decorate the training step for graph execution
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Benchmark loop
num_iterations = 100  # Number of training iterations for the benchmark
warmup_iterations = 10 # Number of iterations to warmup before timing
times = []

print("Starting benchmark...")

for i in range(num_iterations + warmup_iterations):
    start_time = time.time()
    loss = train_step(input_tensor, output_tensor)  # Execute the training step
    end_time = time.time()

    if i >= warmup_iterations: # Start timing after warmup
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    if (i+1) % 10 == 0:  # Print progress every 10 iterations
        print(f"Iteration {i+1}: Loss = {loss.numpy():.4f}")


# Calculate and print statistics
average_time = np.mean(times)
std_dev_time = np.std(times)

print(f"\nBenchmark Results:")
print(f"Average time per iteration: {average_time:.4f} seconds")
print(f"Standard deviation: {std_dev_time:.4f} seconds")
print(f"Total iterations: {num_iterations}")


# Example of how to get throughput (samples/second)
throughput = num_samples / average_time
print(f"Throughput: {throughput:.2f} samples/second")


# Save the model after benchmarking if needed
# model.save("my_benchmarked_model")

print("Benchmark complete.")
