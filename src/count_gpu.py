import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print(f"Is GPU available: {tf.test.is_gpu_available()}")
gpu_list = get_available_gpus()
print(f"Count GPU: {gpu_list}")
