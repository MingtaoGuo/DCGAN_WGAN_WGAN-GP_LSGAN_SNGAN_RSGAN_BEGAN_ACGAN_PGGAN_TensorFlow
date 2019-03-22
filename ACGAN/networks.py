import tensorflow as tf
from ops import *


class generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, z):
        with tf.variable_scope(self.name):
            nums_in = z.shape[-1]
            inputs = relu(linear("projection", z, nums_in, 4*4*512))
            inputs = tf.reshape(inputs, [-1, 4, 4, 512])
            inputs = relu(instance_norm("IN1", deconv("deconv1", inputs, 512, 5, 2)))
            inputs = relu(instance_norm("IN2", deconv("deconv2", inputs, 256, 5, 2)))
            inputs = relu(instance_norm("IN3", deconv("deconv3", inputs, 128, 5, 2)))
            inputs = relu(instance_norm("IN4", deconv("deconv4", inputs, 64, 5, 2)))
            inputs = tanh(conv("conv4", inputs, 3, 5, 1))
        return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, nums_class):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = leaky_relu(conv("conv1", inputs, 64, 5, 2))
            inputs = leaky_relu(instance_norm("IN1", conv("conv2", inputs, 128, 5, 2)))
            inputs = leaky_relu(instance_norm("IN2", conv("conv3", inputs, 256, 5, 2)))
            inputs = leaky_relu(instance_norm("in3", conv("conv4", inputs, 512, 5, 2)))
            logits_adv = linear("logits_adv", inputs, 4 * 4 * 512, 1)
            logits_cls = linear("logits_cls", inputs, 4 * 4 * 512, nums_class)
        return tf.nn.softmax(logits_cls), tf.nn.sigmoid(logits_adv)

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)




