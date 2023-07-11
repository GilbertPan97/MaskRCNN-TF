import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'
import tensorflow as tf
tf.test.is_gpu_available()
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
tf.config.list_physical_devices('GPU')
from tensorflow.python.client import device_lib
device_lib.list_local_devices()