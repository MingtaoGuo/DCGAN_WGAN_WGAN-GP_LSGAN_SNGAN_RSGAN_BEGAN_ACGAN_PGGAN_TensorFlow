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

def read_face_data(path):
    mat_data = sio.loadmat(path)
    data, label = mat_data["data"], mat_data["label"]
    return data, label

def get_batch_face(data, labels, batchsize):
    labels = labels[0, :]
    nums = int(data.shape[0])
    rand_select = np.random.randint(0, nums, [batchsize])
    batch = data[rand_select]
    labels = labels[rand_select]
    z = np.random.normal(0, 1, [batchsize, 100])
    return batch, labels, z

def get_batch(data, labels, batchsize):
    rand_select = np.random.randint(0, 50000, [batchsize])
    batch = data[rand_select]
    labels = labels[rand_select, 0]
    z = np.random.normal(0, 1, [batchsize, 100])
    return batch, labels, z

# a, b = read_data("./dataset/")
# a = 0