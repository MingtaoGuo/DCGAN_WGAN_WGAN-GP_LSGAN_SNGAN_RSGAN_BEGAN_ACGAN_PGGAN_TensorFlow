import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io as sio
from network import generator, discriminator
from ops import l1_loss


BATCH_SIZE = 16
LEARNING_RATE = 1e-4
h = 64
n = 64#paper: 128
LAMBDA = 0.001
GAMMA = 0.4
IMG_SIZE = 64

def Main():
    real_img = tf.placeholder("float", [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    z = tf.placeholder("float", [BATCH_SIZE, h])
    G = generator("generator")
    D = discriminator("discriminator")
    k_t = tf.get_variable("k", initializer=[0.])
    fake_img = G(z, IMG_SIZE, n)
    real_logits = D(real_img, IMG_SIZE, n, h)
    fake_logits = D(fake_img, IMG_SIZE, n, h)
    real_loss = l1_loss(real_img, real_logits)
    fake_loss = l1_loss(fake_img, fake_logits)
    D_loss = real_loss - k_t * fake_loss
    G_loss = fake_loss
    M_global = real_loss + tf.abs(GAMMA * real_loss - fake_loss)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, 5000, 0.5)
    Opt_D = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D.var(), global_step=global_step)
    Opt_G = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G.var())
    with tf.control_dependencies([Opt_D, Opt_G]):
        clip = tf.clip_by_value(k_t + LAMBDA * (GAMMA * real_loss - fake_loss), 0, 1)
        update_k = tf.assign(k_t, clip)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        facedata = sio.loadmat("../TrainingSet/facedata.mat")["data"]
        saver = tf.train.Saver()
        # saver.restore(sess, "./save_para/.\\model.ckpt")
        for epoch in range(200):
            for i in range(facedata.shape[0]//BATCH_SIZE - 1):
                batch = facedata[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE, :, :, :] / 127.5 - 1.0
                z0 = np.random.uniform(0, 1, [BATCH_SIZE, h])
                sess.run(update_k, feed_dict={real_img: batch, z: z0})
                if i % 100 == 0:
                    [dloss, gloss, Mglobal, fakeimg, step, lr] = sess.run([D_loss, G_loss, M_global, fake_img, global_step, learning_rate], feed_dict={real_img: batch, z: z0})
                    print("step: %d, d_loss: %f, g_loss: %f, M_global: %f, Learning_rate: %f"%(step, dloss, gloss, Mglobal, lr))
                    Image.fromarray(np.uint8(127.5*(fakeimg[0, :, :, :]+1))).save("./Results/"+str(step)+".jpg")
            saver.save(sess, "./save_para/model.ckpt")

if __name__ == "__main__":
    Main()