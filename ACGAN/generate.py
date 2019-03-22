from networks import generator
import tensorflow as tf
from PIL import Image
import numpy as np
import os


BATCHSIZE = 64
H = 64
W = 64
NUMS_CLASS = 2

def label_from_0_to_1():
    Z = np.random.normal(0, 1, [1, 100])
    label0 = np.concatenate((np.ones([1, 1]), np.zeros([1, 1])), axis=1)
    label1 = np.concatenate((np.zeros([1, 1]), np.ones([1, 1])), axis=1)
    label = np.zeros([10, 2])
    z = np.zeros([10, 100])
    for i in range(10):
        label[i, :] = label0 + (label1 - label0) * i / 9
        z[i, :] = Z
    return label, z

def from_noise0_to_noise1():
    noise0 = np.random.normal(0, 1, [1, 100])
    noise1 = np.random.normal(0, 1, [1, 100])
    noise = np.zeros([10, 100])
    for i in range(10):
        noise[i, :] = noise0 + (noise1 - noise0) * i / 9
    return noise

def generate_fixed_z():
    label = tf.placeholder(tf.float32, [None, NUMS_CLASS])
    z = tf.placeholder(tf.float32, [None, 100])
    labeled_z = tf.concat([z, label], axis=1)
    G = generator("generator")
    fake_img = G(labeled_z)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/model.ckpt")


    LABELS, Z = label_from_0_to_1()
    if not os.path.exists("./generate_fixed_noise"):
        os.mkdir("./generate_fixed_noise")
    FAKE_IMG = sess.run(fake_img, feed_dict={label: LABELS, z: Z})
    for i in range(10):
        Image.fromarray(np.uint8((FAKE_IMG[i, :, :, :] + 1) * 127.5)).save("./generate_fixed_noise/" + str(i) + ".jpg")


def generate_fixed_label():
    label = tf.placeholder(tf.int32, [None])
    z = tf.placeholder(tf.float32, [None, 100])
    one_hot_label = tf.one_hot(label, NUMS_CLASS)
    labeled_z = tf.concat([z, one_hot_label], axis=1)
    G = generator("generator")
    fake_img = G(labeled_z)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/model.ckpt")

    Z = from_noise0_to_noise1()
    LABELS = np.ones([10])#woman: LABELS = np.ones([10]), man: LABELS = np.zeros([10])
    if not os.path.exists("./generate_fixed_label"):
        os.mkdir("./generate_fixed_label")
    FAKE_IMG = sess.run(fake_img, feed_dict={label: LABELS, z: Z})
    for i in range(10):
        Image.fromarray(np.uint8((FAKE_IMG[i, :, :, :] + 1) * 127.5)).save("./generate_fixed_label/" + str(i) + "_" + str(int(LABELS[i])) + ".jpg")


if __name__ == "__main__":
    #generate_fixed_z() -> fixed the noise and change the label
    #generate_fixed_label() -> fixed the label and change the noise
    generate_fixed_z()