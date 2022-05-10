import tensorflow as tf
from  tensorflow import  keras
print(keras.__version__)
print(tf.__version__)
import tensorflow as tf
print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())
import tensorflow as tf
version=tf.__version__  #输出tensorflow版本
gpu_ok=tf.test.is_gpu_available()  #输出gpu可否使用（True/False）
print("tf version:",version,"\nuse GPU:",gpu_ok)
print(tf.test.is_built_with_cuda() ) # 判断CUDA是否可用（True/False）
