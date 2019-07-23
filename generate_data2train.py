import scipy.io as sio
from PIL import Image
import numpy as np
import os


IMG_PATH = "./img_file/"  #Please create the folder 'img_file' and put all images, which are used to train, into this folder.
SAVE_PATH = "./TrainingSet/"
IMG_H = 64
IMG_W = 64

def generate():
    img_names = os.listdir(IMG_PATH)
    img_num = img_names.__len__()
    data = np.zeros([img_num, IMG_H, IMG_W, 3])
    for idx, img_name in enumerate(img_names):
        img = np.array(Image.open(IMG_PATH + img_name).resize([IMG_W, IMG_H]))
        shape = img.shape
        shape_len = shape.__len__()
        if shape_len < 3:
            img = np.dstack((img, img, img))
        else:
            img = img[:, :, :3]
        data[idx, :, :, :] = img
        print("Total: %d, Progress: %d"%(img_num, idx))
    sio.savemat(SAVE_PATH, {"data": data})


if __name__ == "__main__":
    generate()