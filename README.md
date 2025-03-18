# TensorSlow

Here is just where I keep my Machine Learning study materials.

## Install essential Python packages

1. Activate VirtualEnv: `source bin/activate`
1. `pip install -r requirements.txt`
1. `ansible-playbook ansible_playbooks/install_essential_python_packages_and_utilities.yml`


## CPU accelerated TensorFlow

1. Activate VirtualEnv: `source bin/activate`
1. Build CPU accelerated TensorFlow wheel package: `ansible-playbook --ask-become-pass ansible_playbooks/build_cpu_accelerated_tensorflow_wheel_package.yml`
1. Install the wheel package: `pip install tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp312-cp312-linux_x86_64.whl`
