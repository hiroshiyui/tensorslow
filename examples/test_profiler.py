#!/usr/bin/env python

import tensorflow as tf

def my_function():
    # ... your TensorFlow code ...

    # Wrap sections of code you want to profile with tf.profiler.experimental.Trace
    with tf.profiler.experimental.Trace("my_important_section"):
        # Code within this block will be profiled in detail
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        tf.print(c)

    # Another traced section
    with tf.profiler.experimental.Trace("another_section"):
        d = tf.multiply(a, b)
        tf.print(d)

    # ... more of your code ...


# Enable the profiler.  It's best to do this before the code you want to profile.
log_dir = "logs/my_profiling_data"  # Directory to store profiling data
tf.profiler.experimental.start(log_dir)

my_function()  # Run your TensorFlow code

tf.profiler.experimental.stop()  # Stop the profiler after your code runs
