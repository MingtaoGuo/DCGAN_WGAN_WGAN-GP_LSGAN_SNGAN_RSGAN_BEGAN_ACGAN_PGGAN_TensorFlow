from networks import generator, discriminator
import tensorflow as tf
from utils import read_face_data, get_batch_face, read_data, get_batch
from PIL import Image
import numpy as np
import time


EPSILON = 1e-14
BATCHSIZE = 64
H = 64
W = 64
NUMS_CLASS = 2

def train():
    real_img = tf.placeholder(tf.float32, [None, H, W, 3])
    label = tf.placeholder(tf.int32, [None])
    z = tf.placeholder(tf.float32, [None, 100])
    one_hot_label = tf.one_hot(label, NUMS_CLASS)
    labeled_z = tf.concat([z, one_hot_label], axis=1)
    G = generator("generator")
    D = discriminator("discriminator")
    fake_img = G(labeled_z)
    class_fake_logits, adv_fake_logits = D(fake_img, NUMS_CLASS)
    class_real_logits, adv_real_logits = D(real_img, NUMS_CLASS)
    loss_d_real = -tf.reduce_mean(tf.log(adv_real_logits + EPSILON))
    loss_d_fake = -tf.reduce_mean(tf.log(1 - adv_fake_logits + EPSILON))
    loss_cls_real = -tf.reduce_mean(tf.log(tf.reduce_sum(class_real_logits * one_hot_label, axis=1) + EPSILON))
    loss_cls_fake = -tf.reduce_mean(tf.log(tf.reduce_sum(class_fake_logits * one_hot_label, axis=1) + EPSILON))
    D_loss = loss_d_real + loss_d_fake + loss_cls_real
    G_loss =  -tf.reduce_mean(tf.log(adv_fake_logits + EPSILON)) + loss_cls_fake

    D_opt = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(D_loss, var_list=D.var_list())
    G_opt = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(G_loss, var_list=G.var_list())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    data, labels = read_face_data("./dataset/face_woman_man.mat")
    for i in range(50000):
        s = time.time()
        for j in range(1):
            BATCH, LABELS, Z = get_batch_face(data, labels, BATCHSIZE)
            BATCH = BATCH / 127.5 - 1.0
            sess.run(D_opt, feed_dict={real_img: BATCH, label: LABELS, z: Z})
        sess.run(G_opt, feed_dict={real_img: BATCH, label: LABELS, z: Z})
        e = time.time()
        if i % 10 == 0:
            [D_LOSS, G_LOSS, FAKE_IMG] = sess.run([D_loss, G_loss, fake_img], feed_dict={real_img: BATCH, label: LABELS, z: Z})
            Image.fromarray(np.uint8((FAKE_IMG[0, :, :, :] + 1) * 127.5)).save("./results/" + str(i) +"_" + str(int(LABELS[0])) + ".jpg")
            print("Iteration: %d, D_loss: %f, G_loss: %f, update_time: %f"%(i, D_LOSS, G_LOSS, e-s))
        if i % 500 == 0:
            saver.save(sess, "./save_para/model.ckpt")
    pass

if __name__ == "__main__":
    train()
