import scipy.io as sio
import numpy as np



def read_data(path):
    for i in range(1, 6):
        if i == 1:
            data_mat = sio.loadmat(path + "data_batch_" + str(i) + ".mat")
            data = np.transpose(np.reshape(data_mat["data"], [10000, 3, 32, 32]), [0, 2, 3, 1])
            labels = data_mat["labels"]
        else:
            data_mat = sio.loadmat(path + "data_batch_" + str(i) + ".mat")
            temp = np.transpose(np.reshape(data_mat["data"], [10000, 3, 32, 32]), [0, 2, 3, 1])
            data = np.concatenate((temp, data), axis=0)
            labels = np.concatenate((data_mat["labels"], labels), axis=0)
    return data, labels


def get_batch(data, batchsize):
    data_nums = data.shape[0]
    rand_select = np.random.randint(0, data_nums, [batchsize])
    batch = data[rand_select]
    z = np.random.normal(0, 1, [batchsize, 512])
    return batch, z

def read_face_data(path):
    data = sio.loadmat(path)
    return data["data"]

# a, b = read_data("./dataset/")
# a = 0
