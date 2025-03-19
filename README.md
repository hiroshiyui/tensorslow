# TensorSlow

This repository contains my Machine Learning study materials.

## Setting Up the Development Environment

This section outlines the steps to create and configure a virtual environment for this project.

### Creating a Virtual Environment

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hiroshiyui/tensorslow.git
    ```
2.  **Create the virtual environment:**
    ```bash
    virtualenv --python=python3.12 ./tensorslow
    ```

### Installing Required Python Packages

**Steps:**

1.  **Activate the virtual environment:**
    ```bash
    cd $HOME/tensorslow && source ./bin/activate
    ```
2.  **Install Python package requirements:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Ansible Playbook for additional packages:**
    ```bash
    ansible-playbook ansible_playbooks/install_essential_python_packages_and_utilities.yml
    ```

## CPU-Accelerated TensorFlow (Considerations)

Building a CPU-accelerated TensorFlow package may not provide significant performance gains. For substantial performance improvements, consider using platforms with GPU acceleration, such as [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/).

### Building and Installing CPU-Accelerated TensorFlow (Optional)

**Steps:**

1.  **Activate the virtual environment:**
    ```bash
    cd $HOME/tensorslow && source ./bin/activate
    ```
2.  **Build the CPU-accelerated TensorFlow wheel package (requires sudo):**
    ```bash
    ansible-playbook --ask-become-pass ansible_playbooks/build_cpu_accelerated_tensorflow_wheel_package.yml
    ```
3.  **Install the built wheel package:**
    ```bash
    pip install tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl
    ```
