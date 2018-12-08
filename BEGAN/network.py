import tensorflow as tf
from ops import *

def Encoder(inputs, n=128, h=64):
    with tf.variable_scope("Encoder"):
        inputs = elu(conv("conv1", inputs, n, 3, 1))
        inputs = elu(conv("conv2_1", inputs, n, 3, 1))
        inputs = elu(conv("conv2_2", inputs, 2 * n, 3, 1))
        inputs = Subsampling(inputs)
        inputs = elu(conv("conv3_1", inputs, 2 * n, 3, 1))
        inputs = elu(conv("conv3_2", inputs, 3 * n, 3, 1))
        inputs = Subsampling(inputs)
        inputs = elu(conv("conv4_1", inputs, 3 * n, 3, 1))
        inputs = elu(conv("conv4_2", inputs, 3 * n, 3, 1))
        inputs = fully_connected("Embedding", inputs, h)
    return inputs

def Decoder(inputs, img_size, n=128):
    down_size = img_size // 4
    with tf.variable_scope("Decoder"):
        inputs = fully_connected("Decoder", inputs, down_size * down_size * 3 * n)
        inputs = tf.reshape(inputs, [-1, down_size, down_size, 3*n])
        inputs = elu(conv("conv1_1", inputs, 3 * n, 3, 1))
        inputs = elu(conv("conv1_2", inputs, 3 * n, 3, 1))
        inputs = NN_Upsampling(inputs, down_size * 2)
        inputs = elu(conv("conv2_1", inputs, 3 * n, 3, 1))
        inputs = elu(conv("conv2_2", inputs, 2 * n, 3, 1))
        inputs = NN_Upsampling(inputs, down_size * 4)
        inputs = elu(conv("conv3_1", inputs, 2 * n, 3, 1))
        inputs = elu(conv("conv3_2", inputs, n, 3, 1))
        inputs = conv("conv3", inputs, 3, 3, 1)
    return tf.nn.tanh(inputs)

class generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, img_size, n):
        with tf.variable_scope(self.name):
            inputs = Decoder(inputs, img_size, n)
        return inputs

    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, img_size, n, h):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = Encoder(inputs, n, h)
            inputs = Decoder(inputs, img_size, n)
        return inputs

    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)