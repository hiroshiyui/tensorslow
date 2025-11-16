# TensorSlow

This repository contains my Machine Learning study materials.

## Setting Up the Development Environment

This section outlines the steps to create and configure a virtual environment for this project.

### Installing Required Python Packages

**Steps:**

1.  **Activate the virtual environment:**
    ```bash
    cd $HOME/tensorslow && source ./virtualenv/bin/activate
    ```
2.  **Specify temporary directory in order to avoid <code>/tmp</code> being used:**
    ```bash
    mkdir -p $PWD/tmp && export TMPDIR=$PWD/tmp
    ```
3.  **Install Python package requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## CPU-Accelerated TensorFlow (Considerations)

Building a CPU-accelerated TensorFlow package may not provide significant performance gains. For substantial performance improvements, consider using platforms with GPU acceleration, such as [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/).

### Building and Installing CPU-Accelerated TensorFlow (Optional)

**Steps:**

1.  **Activate the virtual environment:**
    ```bash
    cd $HOME/tensorslow && source ./virtualenv/bin/activate
    ```
2.  **Build the CPU-accelerated TensorFlow wheel package (requires sudo):**
    ```bash
    ansible-playbook --ask-become-pass ansible_playbooks/build_cpu_accelerated_tensorflow_wheel_package.yml
    ```
3.  **Install the built wheel package:**
    ```bash
    pip install tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl
    ```
