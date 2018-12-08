import tensorflow as tf
import numpy as np

def conv(name, inputs, nums_out, ksize, strides, padding="SAME"):
    nums_in = int(inputs.shape[-1])
    W = tf.get_variable("W"+name, [ksize, ksize, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable("b"+name, [nums_out], initializer=tf.constant_initializer(0.))
    return tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding) + b


def fully_connected(name, inputs, nums_out):
    inputs = tf.layers.flatten(inputs)
    nums_in = inputs.shape[-1]
    W = tf.get_variable("W" + name, [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable("b" + name, [nums_out], initializer=tf.truncated_normal_initializer(stddev=0.))
    return tf.matmul(inputs, W) + b

def elu(inputs):
    return tf.nn.elu(inputs)

def Subsampling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def NN_Upsampling(inputs, out_size):
    return tf.image.resize_nearest_neighbor(inputs, [out_size, out_size])


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))
