from ops import *

class generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, alpha=None, dim=[], pg=0):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = fully_connected("projection", inputs, 4*4*512)
            inputs = tf.reshape(inputs, [-1, 4, 4, 512])
            inputs = G_first_block("block_first", inputs, dim[0])
            if pg == 1:
                upscaled = upsample(inputs)
                upscaled_rgb1 = to_RGB("toRGB1_"+str(pg), upscaled)
                upscaled2 = G_block("block_"+str(pg), upscaled, dim[pg])
                upscaled_rgb2 = to_RGB("toRGB2_"+str(pg), upscaled2)
                inputs = (1 - alpha) * upscaled_rgb1 + alpha * upscaled_rgb2
            for i in range(1, pg):
                inputs = upsample(inputs)
                inputs = G_block("block_"+str(i), inputs, dim[i])
            if pg > 1:
                upscaled = upsample(inputs)
                upscaled_rgb1 = to_RGB("toRGB1_"+str(pg), upscaled)
                upscaled2 = G_block("block_"+str(pg), upscaled, dim[pg])
                upscaled_rgb2 = to_RGB("toRGB2_"+str(pg), upscaled2)
                inputs = (1 - alpha) * upscaled_rgb1 + alpha * upscaled_rgb2
            elif pg == 0:
                inputs = to_RGB("toRGB", inputs)
        return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, alpha=None, dim=[], pg=0):
        layer_size = dim.__len__()
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if pg == 0:
                inputs = from_RGB("fromRGB", inputs, dim[layer_size-2])
            else:
                inputs = D_transition("block_" + str(layer_size-pg-1), inputs, dim[layer_size-pg-2], dim[layer_size-pg-1], alpha)
            for i in range(layer_size-pg, layer_size-1):
                inputs = D_block("block_"+str(i), inputs, dim[i])
                inputs = downsample(inputs)
            inputs = D_last_block("block_last", inputs, dim[0])
        return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)








