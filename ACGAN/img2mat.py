import scipy.io as sio
from PIL import Image
import numpy as np
import os

#F:\BaiduNetdiskDownload\man2woman\a_resized
def img2mat(path_man, path_woman):
    filenames_man = os.listdir(path_man)
    filenames_woman = os.listdir(path_woman)
    man_mat = np.zeros([filenames_man.__len__(), 64, 64, 3], dtype=np.uint8)
    man_label = np.zeros([filenames_man.__len__()])
    woman_mat = np.zeros([filenames_woman.__len__(), 64, 64, 3], dtype=np.uint8)
    woman_label = np.ones([filenames_woman.__len__()])
    for idx, filename_man in enumerate(filenames_man):
        img = np.array(Image.open(path_man + filename_man).resize([64, 64]))
        man_mat[idx, :, :, :] = img
    for idx, filename_woman in enumerate(filenames_woman):
        img = np.array(Image.open(path_woman + filename_woman).resize([64, 64]))
        woman_mat[idx, :, :, :] = img
    data = np.concatenate((man_mat, woman_mat), axis=0)
    label = np.concatenate((man_label, woman_label), axis=0)
    sio.savemat("face_woman_man.mat", {"data": data, "label": label})

if __name__ == "__main__":
    img2mat("F://BaiduNetdiskDownload//man2woman//a_resized//", "F://BaiduNetdiskDownload//man2woman//b_resized//")


