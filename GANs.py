import tensorflow as tf
import scipy.io as sio
import numpy as np
from PIL import Image

img_H = 64
img_W = 64
img_C = 3
GAN_type = "RSGAN"  # DCGAN, WGAN, WGAN-GP, SNGAN, LSGAN, RSGAN, RaSGAN
batchsize = 128
epsilon = 1e-14#if epsilon is too big, training of DCGAN is failure.

def deconv(inputs, shape, strides, out_num, is_sn=False):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-2]], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.nn.conv2d_transpose(inputs, spectral_norm("sn", filters), out_num, strides) + bias
    else:
        return tf.nn.conv2d_transpose(inputs, filters, out_num, strides) + bias

def conv(inputs, shape, strides, is_sn=False):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-1]], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.nn.conv2d(inputs, spectral_norm("sn", filters), strides, "SAME") + bias
    else:
        return tf.nn.conv2d(inputs, filters, strides, "SAME") + bias

def fully_connected(inputs, num_out, is_sn=False):
    W = tf.get_variable("W", [inputs.shape[-1], num_out], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0]))
    if is_sn:
        return tf.matmul(inputs, spectral_norm("sn", W)) + b
    else:
        return tf.matmul(inputs, W) + b

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope*inputs, inputs)

def spectral_norm(name, w, iteration=1):
    #Spectral normalization which was published on ICLR2018,please refer to "https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks"
    #This function spectral_norm is forked from "https://github.com/taki0112/Spectral_Normalization-Tensorflow"
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm

def mapping(x):
    max = np.max(x)
    min = np.min(x)
    return (x - min) * 255.0 / (max - min + epsilon)

def instanceNorm(inputs):
    mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    scale = tf.get_variable("scale", shape=mean.shape, initializer=tf.constant_initializer([1.0]))
    shift = tf.get_variable("shift", shape=mean.shape, initializer=tf.constant_initializer([0.0]))
    return (inputs - mean) * scale / (tf.sqrt(var + epsilon)) + shift

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, Z):
        with tf.variable_scope(name_or_scope=self.name, reuse=False):
            with tf.variable_scope(name_or_scope="linear"):
                inputs = tf.reshape(tf.nn.relu((fully_connected(Z, 4*4*512))), [batchsize, 4, 4, 512])
            with tf.variable_scope(name_or_scope="deconv1"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 256, 512], [1, 2, 2, 1], [batchsize, 8, 8, 256])))
            with tf.variable_scope(name_or_scope="deconv2"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 128, 256], [1, 2, 2, 1], [batchsize, 16, 16, 128])))
            with tf.variable_scope(name_or_scope="deconv3"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 64, 128], [1, 2, 2, 1], [batchsize, 32, 32, 64])))
            if img_H == 32:
                with tf.variable_scope(name_or_scope="deconv4"):
                    inputs = tf.nn.tanh(deconv(inputs, [5, 5, img_C, 64], [1, 1, 1, 1], [batchsize, img_H, img_W, img_C]))
                return inputs
            with tf.variable_scope(name_or_scope="deconv4"):
                 inputs = tf.nn.tanh(deconv(inputs, [5, 5, img_C, 64], [1, 2, 2, 1], [batchsize, img_H, img_W, img_C]))
            if img_H == 64:
                return inputs

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, reuse=False, is_sn=False):
        with tf.variable_scope(name_or_scope=self.name, reuse=reuse):
            with tf.variable_scope("conv1"):
                inputs = leaky_relu(conv(inputs, [5, 5, img_C, 64], [1, 2, 2, 1], is_sn))
            with tf.variable_scope("conv2"):
                inputs = leaky_relu(instanceNorm(conv(inputs, [5, 5, 64, 128], [1, 2, 2, 1], is_sn)))
            with tf.variable_scope("conv3"):
                inputs = leaky_relu(instanceNorm(conv(inputs, [5, 5, 128, 256], [1, 2, 2, 1], is_sn)))
            with tf.variable_scope("conv4"):
                inputs = leaky_relu(instanceNorm(conv(inputs, [5, 5, 256, 512], [1, 2, 2, 1], is_sn)))
            inputs = tf.layers.flatten(inputs)
            return fully_connected(inputs, 1, is_sn)

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class GAN:
    #Architecture of generator and discriminator just like DCGAN.
    def __init__(self):
        self.Z = tf.placeholder("float", [batchsize, 100])
        self.img = tf.placeholder("float", [batchsize, img_H, img_W, img_C])
        D = Discriminator("discriminator")
        G = Generator("generator")
        self.fake_img = G(self.Z)
        if GAN_type == "DCGAN":
            #DCGAN, paper: UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
            self.fake_logit = tf.nn.sigmoid(D(self.fake_img))
            self.real_logit = tf.nn.sigmoid(D(self.img, reuse=True))
            self.d_loss = - (tf.reduce_mean(tf.log(self.real_logit + epsilon)) + tf.reduce_mean(tf.log(1 - self.fake_logit + epsilon)))
            self.g_loss = - tf.reduce_mean(tf.log(self.fake_logit + epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif GAN_type == "WGAN":
            #WGAN, paper: Wasserstein GAN
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.d_loss = -tf.reduce_mean(self.real_logit) + tf.reduce_mean(self.fake_logit)
            self.g_loss = -tf.reduce_mean(self.fake_logit)
            self.clip = []
            for _, var in enumerate(D.var):
                self.clip.append(tf.clip_by_value(var, -0.01, 0.01))
            self.opt_D = tf.train.RMSPropOptimizer(5e-5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.RMSPropOptimizer(5e-5).minimize(self.g_loss, var_list=G.var)
        elif GAN_type == "WGAN-GP":
            #WGAN-GP, paper: Improved Training of Wasserstein GANs
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            e = tf.random_uniform([batchsize, 1, 1, 1], 0, 1)
            x_hat = e * self.img + (1 - e) * self.fake_img
            grad = tf.gradients(D(x_hat, reuse=True), x_hat)[0]
            self.d_loss = tf.reduce_mean(self.fake_logit - self.real_logit) + 10 * tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])) - 1))
            self.g_loss = tf.reduce_mean(-self.fake_logit)
            self.opt_D = tf.train.AdamOptimizer(1e-4, beta1=0., beta2=0.9).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(1e-4, beta1=0., beta2=0.9).minimize(self.g_loss, var_list=G.var)
        elif GAN_type == "LSGAN":
            #LSGAN, paper: Least Squares Generative Adversarial Networks
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.d_loss = tf.reduce_mean(0.5 * tf.square(self.real_logit - 1) + 0.5 * tf.square(self.fake_logit))
            self.g_loss = tf.reduce_mean(0.5 * tf.square(self.fake_logit - 1))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif GAN_type == "SNGAN":
            #SNGAN, paper: SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS
            self.fake_logit = tf.nn.sigmoid(D(self.fake_img, is_sn=True))
            self.real_logit = tf.nn.sigmoid(D(self.img, reuse=True, is_sn=True))
            self.d_loss = - (tf.reduce_mean(tf.log(self.real_logit + epsilon) + tf.log(1 - self.fake_logit + epsilon)))
            self.g_loss = - tf.reduce_mean(tf.log(self.fake_logit + epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif GAN_type == "RSGAN":
            #RSGAN, paper: The relativistic discriminator: a key element missing from standard GAN
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.d_loss = - tf.reduce_mean(tf.log(tf.nn.sigmoid(self.real_logit - self.fake_logit) + epsilon))
            self.g_loss = - tf.reduce_mean(tf.log(tf.nn.sigmoid(self.fake_logit - self.real_logit) + epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif GAN_type == "RaSGAN":
            #RaSGAN, paper: The relativistic discriminator: a key element missing from standard GAN
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.avg_fake_logit = tf.reduce_mean(self.fake_logit)
            self.avg_real_logit = tf.reduce_mean(self.real_logit)
            self.D_r_tilde = tf.nn.sigmoid(self.real_logit - self.avg_fake_logit)
            self.D_f_tilde = tf.nn.sigmoid(self.fake_logit - self.avg_real_logit)
            self.d_loss = - tf.reduce_mean(tf.log(self.D_r_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - self.D_f_tilde + epsilon))
            self.g_loss = - tf.reduce_mean(tf.log(self.D_f_tilde + epsilon)) - tf.reduce_mean(tf.log(1 - self.D_r_tilde + epsilon))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self):
        saver = tf.train.Saver()
        epoch_nums = 200
        facedata = sio.loadmat("./TrainingSet//facedata.mat")["data"]
        #For face data, i random select about 10,000 images from CelebA and resize them to 64x64 by Matlab.
        for epoch in range(epoch_nums):
            for i in range(facedata.__len__()//batchsize-1):
                batch = facedata[i*batchsize:i*batchsize+batchsize, :, :, :] / 255.0
                z = np.random.standard_normal([batchsize, 100])
                d_loss = self.sess.run(self.d_loss, feed_dict={self.img: batch, self.Z: z})
                g_loss = self.sess.run(self.g_loss, feed_dict={self.img: batch, self.Z: z})
                self.sess.run(self.opt_D, feed_dict={self.img: batch, self.Z: z})
                if GAN_type == "WGAN":
                    self.sess.run(self.clip)#WGAN weight clipping
                self.sess.run(self.opt_G, feed_dict={self.img: batch, self.Z: z})
                if i % 10 == 0:
                    print("epoch: %d, step: %d, d_loss: %g, g_loss: %g"%(epoch, i,  d_loss, g_loss))
                    z = np.random.standard_normal([batchsize, 100])
                    imgs = self.sess.run(self.fake_img, feed_dict={self.img: batch, self.Z: z})
                    for j in range(batchsize):
                        if img_C == 1:
                            Image.fromarray(np.reshape(np.uint8(mapping(imgs[j, :, :, :])), [img_H, img_W])).save(
                                "./result//" + str(epoch) + "_" + str(j) + ".jpg")
                        else:
                            Image.fromarray(np.uint8(mapping(imgs[j, :, :, :]))).save(
                                "./result//" + str(epoch) + "_" + str(j) + ".jpg")
            saver.save(self.sess, "./para//model.ckpt")



if __name__ == "__main__":
    gan = GAN()
    gan()
