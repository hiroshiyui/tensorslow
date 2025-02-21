# TensorSlow

Here is just where I keep my Machine Learning study materials.

## CPU accelerated TensorFlow

* Build CPU accelerated TensorFlow wheel package: `ansible-playbook --ask-become-pass ansible_playbooks/build_cpu_accelerated_tensorflow_wheel_package.yml`
* Install the wheel package: `pip install tensorflow/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.18.0-cp312-cp312-linux_x86_64.whl`
