from networks import generator, discriminator
import tensorflow as tf
from utils import read_face_data, get_batch
from ops import LSGAN_LOSS
from PIL import Image
import numpy as np

BATCH_SIZE = 32
MAX_ITR = 10000
LEARNING_RATE = 1e-3
G_DIM = [512, 256, 128, 64, 32, 16]
D_DIM = [16, 32, 64, 128, 256, 512]

data = read_face_data("./dataset/facedata.mat")


def train():
    G = generator("generator")
    D = discriminator("discriminator")
    inputs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    z = tf.placeholder(tf.float32, [None, 512])
    alpha = tf.placeholder(tf.float32, [1, 1, 1, 1])

    optimizer_D = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0., beta2=0.99)
    optimizer_G = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0., beta2=0.99)
    fake_4 = G(z, alpha, G_DIM, 0)
    fake_4_logits = D(fake_4, alpha, D_DIM, 0)
    real_4_logits = D(tf.image.resize_images(inputs, [4, 4]), alpha, D_DIM, 0)
    D_loss_4, G_loss_4 = LSGAN_LOSS(real_4_logits, fake_4_logits)
    D_Opt_4 = optimizer_D.minimize(D_loss_4, var_list=D.var_list())
    G_Opt_4 = optimizer_G.minimize(G_loss_4, var_list=G.var_list())

    fake_8 = G(z, alpha, G_DIM, 1)
    fake_8_logits = D(fake_8, alpha, D_DIM, 1)
    real_8_logits = D(tf.image.resize_images(inputs, [8, 8]), alpha, D_DIM, 1)
    D_loss_8, G_loss_8 = LSGAN_LOSS(real_8_logits, fake_8_logits)
    D_Opt_8 = optimizer_D.minimize(D_loss_8, var_list=D.var_list())
    G_Opt_8 = optimizer_G.minimize(G_loss_8, var_list=G.var_list())

    fake_16 = G(z, alpha, G_DIM, 2)
    fake_16_logits = D(fake_16, alpha, D_DIM, 2)
    real_16_logits = D(tf.image.resize_images(inputs, [16, 16]), alpha, D_DIM, 2)
    D_loss_16, G_loss_16 = LSGAN_LOSS(real_16_logits, fake_16_logits)
    D_Opt_16 = optimizer_D.minimize(D_loss_16, var_list=D.var_list())
    G_Opt_16 = optimizer_G.minimize(G_loss_16, var_list=G.var_list())

    fake_32 = G(z, alpha, G_DIM, 3)
    fake_32_logits = D(fake_32, alpha, D_DIM, 3)
    real_32_logits = D(tf.image.resize_images(inputs, [32, 32]), alpha, D_DIM, 3)
    D_loss_32, G_loss_32 = LSGAN_LOSS(real_32_logits, fake_32_logits)
    D_Opt_32 = optimizer_D.minimize(D_loss_32, var_list=D.var_list())
    G_Opt_32 = optimizer_G.minimize(G_loss_32, var_list=G.var_list())

    fake_64 = G(z, alpha, G_DIM, 4)
    fake_64_logits = D(fake_64, alpha, D_DIM, 4)
    real_64_logits = D(tf.image.resize_images(inputs, [64, 64]), alpha, D_DIM, 4)
    D_loss_64, G_loss_64 = LSGAN_LOSS(real_64_logits, fake_64_logits)
    D_Opt_64 = optimizer_D.minimize(D_loss_64, var_list=D.var_list())
    G_Opt_64 = optimizer_G.minimize(G_loss_64, var_list=G.var_list())

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
# -----------------------------------------------------phase 4X4------------------------------------------------------------------------------------
    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(i / MAX_ITR, [1, 1, 1, 1])
        sess.run(D_Opt_4, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_4, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_4, G_loss_4, fake_4],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_4x4_" + str(i) + ".jpg")
            print("phase 4x4, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))

    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(1.0, [1, 1, 1, 1])
        sess.run(D_Opt_4, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_4, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_4, G_loss_4, fake_4],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_4x4_" + str(i) + ".jpg")
            print("phase 4x4, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))
    # --------------------------------------------------phase 8x8---------------------------------------------------------------------------------------
    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(i / MAX_ITR, [1, 1, 1, 1])
        sess.run(D_Opt_8, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_8, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_8, G_loss_8, fake_8],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_8x8_" + str(i) + ".jpg")
            print("phase 8x8 transition, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))

    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(1.0, [1, 1, 1, 1])
        sess.run(D_Opt_8, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_8, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_8, G_loss_8, fake_8],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_8x8_" + str(i) + ".jpg")
            print("phase 8x8 transition, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))

    # --------------------------------------------------phase 16x16---------------------------------------------------------------------------------------

    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(i / MAX_ITR, [1, 1, 1, 1])
        sess.run(D_Opt_16, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_16, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_16, G_loss_16, fake_16],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_16x16_" + str(i) + ".jpg")
            print("phase 16x16, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))

    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(1.0, [1, 1, 1, 1])
        sess.run(D_Opt_16, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_16, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_16, G_loss_16, fake_16],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_16x16_" + str(i) + ".jpg")
            print("phase 16x16, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))

    # --------------------------------------------------phase 32x32---------------------------------------------------------------------------------------

    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(i / MAX_ITR, [1, 1, 1, 1])
        sess.run(D_Opt_32, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_32, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_32, G_loss_32, fake_32],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_32x32_" + str(i) + ".jpg")
            print("phase 32x32 transition, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))

    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(1.0, [1, 1, 1, 1])
        sess.run(D_Opt_32, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_32, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_32, G_loss_32, fake_32],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_32x32_" + str(i) + ".jpg")
            print("phase 32x32 transition, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))

    # --------------------------------------------------phase 64x64---------------------------------------------------------------------------------------

    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(i / MAX_ITR, [1, 1, 1, 1])
        sess.run(D_Opt_64, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_64, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_64, G_loss_64, fake_64],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_64x64_" + str(i) + ".jpg")
            print("phase 64x64 transition, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))

    for i in range(MAX_ITR):
        batch, Z = get_batch(data, BATCH_SIZE)
        ALPHA = np.reshape(1.0, [1, 1, 1, 1])
        sess.run(D_Opt_64, feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
        batch, Z = get_batch(data, BATCH_SIZE)
        sess.run(G_Opt_64, feed_dict={z: Z, alpha: ALPHA})
        if i % 100 == 0:
            [D_LOSS, G_LOSS, FAKE_IMGS] = sess.run([D_loss_64, G_loss_64, fake_64],
                                                   feed_dict={z: Z, inputs: batch / 127.5 - 1, alpha: ALPHA})
            Image.fromarray(np.uint8((FAKE_IMGS[0, :, :, :] + 1) * 127.5)).save(
                "./results/" + "phase_64x64_" + str(i) + ".jpg")
            print("phase 64x64 transition, iteration: %d, D_loss: %f, G_loss: %f" % (i, D_LOSS, G_LOSS))


if __name__ == "__main__":
    train()
