import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

dataset = sio.loadmat('data_batch_4.mat')
data = np.float32(np.array(dataset['data'])) / np.max(np.float32(np.array(dataset['data'])))
labels_1 = np.array(dataset['labels'])
label_no_onehot = []
for j in range(10000):
    label_no_onehot.append(labels_1[j][0])
label_no_onehot = np.array(label_no_onehot)

data_length = np.size(labels_1)
label_max = 10
label = np.zeros([np.size(labels_1),label_max])
for i in range(np.size(labels_1)):
    label[i][labels_1[i]] = 1
data = np.transpose(data) # 3072 * 10000
label = np.transpose(label) # 10000 * 10
def LoadBatch_4():
    return data, label, data_length, label_no_onehot
