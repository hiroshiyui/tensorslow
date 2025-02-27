#!/usr/bin/env python
import tensorflow as tf

def get_cpu_info():
    """Gets and prints CPU information relevant to TensorFlow."""

    cpus = tf.config.list_physical_devices('CPU')

    if cpus:
        print("CPUs detected:")
        for cpu in cpus:
            print(f"  {cpu}")  # Prints the device name (e.g., '/physical_device:CPU:0')

            # The device_details attribute is no longer available.
            # We can get *some* limited info from the device name string.
            # This is not very detailed, but it's what TensorFlow provides directly.
            device_name = cpu.name
            print(f"  Device Name: {device_name}")

            # Example of how to pin operations to a specific CPU if you have multiple
            # with tf.device('/physical_device:CPU:0'):
            #     # Your TensorFlow operations here will run on the first CPU
            #     a = tf.constant([1, 2, 3])
            #     b = tf.constant([4, 5, 6])
            #     c = tf.add(a, b)
            #     print(c)

    else:
        print("No CPUs found by TensorFlow.")

    # For more detailed system CPU info (beyond TensorFlow's scope)
    import platform
    print("\nSystem CPU Information (from platform module):")
    print(f"  Processor: {platform.processor()}") # Often gives a general name, not detailed microarchitecture

    # For even more details, consider the 'psutil' library or OS-specific tools.
    try:
        import psutil
        print("\nSystem CPU Information (from psutil module):")
        cpu_times = psutil.cpu_times()
        print(f"  CPU Times: {cpu_times}") # User, system, idle, etc. times
        print(f"  CPU Count (logical): {psutil.cpu_count()}")
        print(f"  CPU Count (physical): {psutil.cpu_count(logical=False)}") # Number of physical cores
        # ... other psutil CPU info as needed.
    except ImportError:
        print("psutil library not found. Install it for more detailed CPU info (pip install psutil).")



get_cpu_info()
