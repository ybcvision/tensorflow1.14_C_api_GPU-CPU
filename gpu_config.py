import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth = True ,per_process_gpu_memory_fraction=0.10,visible_device_list="0")
config = tf.ConfigProto(gpu_options=gpu_options)
serialized = config.SerializeToString()
list(map(hex,serialized))
print(list(map(hex,serialized)))