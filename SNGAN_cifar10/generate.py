from networks import Generator, Discriminator
from ops import Hinge_loss
import tensorflow as tf
import numpy as np
from utils import random_batch, read_face
from PIL import Image
import time
import scipy.io as sio
import os

NUMS_GEN = 128

def generate():
    train_phase = tf.placeholder(tf.bool)
    z = tf.placeholder(tf.float32, [None, 128])
    G = Generator("generator")
    fake_img = G(z, train_phase)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))
    saver.restore(sess, "./save_para/.\\model.ckpt")
    Z = np.random.standard_normal([NUMS_GEN, 128])
    FAKE_IMG = sess.run(fake_img, feed_dict={z: Z, train_phase: False})
    if not os.path.exists("./generate"):
        os.mkdir("./generate")
    for i in range(NUMS_GEN):
        Image.fromarray(np.uint8((FAKE_IMG[i] + 1) * 127.5)).save("./generate/"+str(i)+".jpg")

if __name__ == "__main__":
    generate()
