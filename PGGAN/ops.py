import tensorflow as tf


EPSILON = 1e-8

def conv(name, inputs, nums_out, ksize, strides, padding="SAME"):
    with tf.variable_scope(name):
        nums_in = int(inputs.shape[-1])
        W = tf.get_variable("W", [ksize, ksize, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer(0.))
    return tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding) + b

def upsample(inputs):
    H = inputs.shape[1]
    W = inputs.shape[2]
    return tf.image.resize_nearest_neighbor(inputs, [H*2, W*2])

def downsample(inputs):
    return tf.nn.avg_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

def fully_connected(name, inputs, nums_out):
    inputs = tf.layers.flatten(inputs)
    nums_in = int(inputs.shape[-1])
    with tf.variable_scope(name):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.truncated_normal_initializer(stddev=0.))
    return tf.matmul(inputs, W) + b

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(inputs, slope * inputs)

def to_RGB(name, inputs, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        inputs = conv("toRGB", inputs, 3, 1, 1)
        inputs = tf.tanh(inputs)
    return inputs

def from_RGB(name, inputs, nums_out):
    with tf.variable_scope(name):
        inputs = leaky_relu(conv("fromRGB", inputs, nums_out, 1, 1))
    return inputs

def G_block(name, inputs, nums_out):
    with tf.variable_scope(name):
        inputs = leaky_relu(lrn(conv("conv1", inputs, nums_out, 4, 1)))
        inputs = leaky_relu(lrn(conv("conv2", inputs, nums_out, 3, 1)))
    return inputs

def G_first_block(name, inputs, nums_out):
    with tf.variable_scope(name):
        inputs = fully_connected("projection", inputs, 4*4*256)
        inputs = tf.reshape(inputs, [-1, 4, 4, 256])
        inputs = leaky_relu(lrn(conv("conv1", inputs, nums_out, 4, 1)))
        inputs = leaky_relu(lrn(conv("conv2", inputs, nums_out, 3, 1)))
    return inputs

def D_block(name, inputs, nums_out):
    with tf.variable_scope(name):
        inputs = leaky_relu(conv("conv1", inputs, nums_out, 3, 1))
        inputs = leaky_relu(conv("conv2", inputs, nums_out, 3, 1))
    return inputs

def D_last_block(name, inputs, nums_out):
    with tf.variable_scope(name):
        inputs = minibatch_stddev(inputs)
        inputs = leaky_relu(conv("conv1", inputs, nums_out, 3, 1))
        inputs = leaky_relu(conv("conv2", inputs, nums_out, 4, 1, "VALID"))
        inputs = fully_connected("logits", inputs, 1)
    return inputs

def D_transition(name, inputs, nums_in, nums_out, alpha):
    y = from_RGB("fromRGB1"+name, inputs, nums_in)
    y = D_block(name, y, nums_out)
    downscaled1 = downsample(y)
    y = downsample(inputs)
    downscaled2 = from_RGB("fromRGB2"+name, y, nums_out)
    inputs = alpha * downscaled1 + (1 - alpha) * downscaled2
    return inputs

def minibatch_stddev(inputs, group_size=4):
    temp = inputs
    b, h, w, c = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
    inputs = tf.reshape(inputs, [group_size, -1, h, w, c])
    inputs = tf.sqrt(tf.reduce_mean(tf.square(inputs - tf.reduce_mean(inputs, axis=0, keep_dims=True)), axis=0) + EPSILON)
    inputs = tf.reduce_mean(inputs, axis=[1, 2, 3], keep_dims=True)
    inputs = tf.tile(inputs, multiples=[group_size, h, w, 1])
    inputs = tf.concat([temp, inputs], axis=-1)
    return inputs


def lrn(inputs):
    return inputs * tf.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keep_dims=True) + 1e-8)

def WGAN_GP_LOSS(real_logits, fake_logits, real_img, fake_img, D, alpha, batchsize):
    e = tf.random_uniform([batchsize, 1, 1, 1], 0, 1)
    x_hat = e * real_img + (1 - e) * fake_img
    grad = tf.gradients(D(x_hat, alpha)[0], x_hat)[0]
    slope = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]) + EPSILON)
    gp = tf.reduce_mean((slope - 1) ** 2)
    d_loss = tf.reduce_mean(fake_logits - real_logits) + 10 * gp
    g_loss = tf.reduce_mean(-fake_logits)
    return d_loss, g_loss

def LSGAN_LOSS(real_logits, fake_logits):
    d_loss = tf.reduce_mean(tf.squared_difference(real_logits, 1.0)) + \
             tf.reduce_mean(tf.square(fake_logits))
    g_loss = tf.reduce_mean(tf.squared_difference(fake_logits, 1.0))
    return d_loss, g_loss


# inputs = tf.placeholder(tf.float32, [64, 32, 32, 128])
# minibatch_stddev(inputs)