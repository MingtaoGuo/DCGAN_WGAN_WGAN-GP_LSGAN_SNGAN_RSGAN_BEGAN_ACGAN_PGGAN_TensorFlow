import tensorflow as tf


def conv(name, inputs, nums_out, k_size, strides, padding="SAME"):
    nums_in = int(inputs.shape[-1])
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
    return tf.nn.bias_add(tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding), b)

def deconv(name, inputs, nums_out, k_size, strides, padding="SAME"):
    nums_in = int(inputs.shape[-1])
    B = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_out, nums_in], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
    return tf.nn.bias_add(tf.nn.conv2d_transpose(inputs, W, output_shape=[B, height*strides, width*strides, nums_out], strides=[1, strides, strides, 1], padding=padding), b)

def linear(name, inputs, nums_in, nums_out):
    inputs = tf.layers.flatten(inputs)
    with tf.variable_scope(name):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
    return tf.nn.bias_add(tf.matmul(inputs, W), b)

def relu(inputs):
    return tf.nn.relu(inputs)

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(inputs, slope * inputs)

def tanh(inputs):
    return tf.nn.tanh(inputs)

def batchnorm(x, train_phase, scope_bn):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def instance_norm(name, inputs):
    with tf.variable_scope(name):
        beta = tf.get_variable("beta", [inputs.shape[-1]], initializer=tf.constant_initializer([0.0]))
        gamma = tf.get_variable("gamma", [inputs.shape[-1]], initializer=tf.constant_initializer([1.0]))
        mu, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    return (inputs - mu) * gamma / tf.sqrt(var + 1e-10) + beta
