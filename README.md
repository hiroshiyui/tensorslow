# TensorSlow

This repository contains my Machine Learning study materials.

## Create a Virtual Environment

1.  **Navigate to Home Directory and Create Project Directory:** `cd $HOME && mkdir -p tensorslow`
2.  **Create Virtual Environment:** `virtualenv --python=python3.12 ./tensorslow`

## Install essential Python packages

1.  **Activate the Virtual Environment:** `source tensorslow/bin/activate`
2.  **Install Python Package Requirements:** `pip install -r requirements.txt`
3.  **Run Ansible Playbook:** `ansible-playbook ansible_playbooks/install_essential_python_packages_and_utilities.yml`

## CPU-Accelerated TensorFlow (Note: Performance gains may be minimal)

It's worth noting that building a CPU-accelerated TensorFlow package might not yield significant performance improvements. For enhanced performance, consider using platforms like [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/), which offer GPU acceleration.

1.  **Activate Virtual Environment:** `source bin/activate`
2.  **Build CPU-Accelerated TensorFlow Wheel Package (Requires sudo):** `ansible-playbook --ask-become-pass ansible_playbooks/build_cpu_accelerated_tensorflow_wheel_package.yml`
3.  **Install the Built Wheel Package:** `pip install tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl`
