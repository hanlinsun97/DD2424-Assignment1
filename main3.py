import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from LoadBatch_1 import LoadBatch_1
from LoadBatch_2 import LoadBatch_2
from initialization import initialization

import scipy.io as sio


#dataset = sio.loadmat('data_batch_1.mat')
#data_test = np.float32(np.array(dataset['data']))
#print(data_test)
#plt.figure(1)
#data_test = np.reshape(data_test[0,:],[32,32,3])
#data_test = data_test.transpose(1,0,2)
#plt.imshow(data_test)
#plt.show()

#Parameter
batch_size = 100
lam = 1
MAX = 40
learning_rate = 0.01
training_data = 10000

# define some functions

def ComputeCost(label, W, lam, batch_size, P):

    Y = label
    loss = np.sum(np.diag(-np.log(np.dot(np.transpose(Y),P))))/batch_size
    #J = loss + lam*(np.sum(-np.log(np.diag(np.dot(np.transpose(W),W)))))
    J = loss + lam*np.sum(np.power(W,2))
    #J = loss+lam*np.sum(W**2)

    return J, loss


def EvaluationClassifier(W, b, input_data):
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)
    s = np.dot(W, input_data) + b

    P = softmax(s)
    return P


def ComputeGradients(W, b, P, input_data, input_label, lam, batch_size):

    # input_data with 3072 * Batch_size
    # input_label with 10 * Batch_size

    grad_W = np.dot((P - input_label), np.transpose(input_data))/batch_size + 2*lam*W
    grad_b = np.mean(P-input_label, 1)
    grad_b = np.reshape(grad_b, [10, 1])
    return grad_W, grad_b


def ComputeAccuracy(P, input_label_no_onehot, batch_size):
    Q = P.argmax(axis=0)  # Predict label
    Y = input_label_no_onehot
    diff = Q - Y
    acc = np.sum(diff == 0)/batch_size
    return acc



#Load data and initialization

[data_1, label_1, data_length_1, label_no_onehot_1] = LoadBatch_1()
[data_2, label_2, data_length_2, label_no_onehot_2] = LoadBatch_2()
[W, b] = initialization()

#Start training!
J_store_1 = []
J_store_2 = []
loss_store_1 = []
loss_store_2 = []
acc_1 = []
acc_2 = []
for epoch in range(MAX):
    print("This is epoch",epoch)

    for i in range(int(training_data/batch_size)):

        input_data = data_1[:,i*batch_size:(i+1)*batch_size]
        input_label = label_1[:,i*batch_size:(i+1)*batch_size]
        input_label_no_onehot = label_no_onehot_1[i*batch_size:(i+1)*batch_size]
        P = EvaluationClassifier(W, b, input_data)  # 10 * Batch_size
        grad_W, grad_b = ComputeGradients(W,b,P,input_data,input_label,lam,batch_size)
        W = W - learning_rate*grad_W
        b = b - learning_rate*grad_b

    P_use = EvaluationClassifier(W, b, data_1)
    J,loss = ComputeCost(label_1, W, lam, data_length_1, P_use)
    acc = ComputeAccuracy(P_use,label_no_onehot_1, data_length_1)
    J_store_1.append(J)
    acc_1.append(acc)
    loss_store_1.append(loss)
# We run our model on validation set


    P_use = EvaluationClassifier(W, b, data_2)
    J,loss = ComputeCost(label_2, W, lam, data_length_2, P_use)
    acc = ComputeAccuracy(P_use, label_no_onehot_2, data_length_2)
    J_store_2.append(J)
    acc_2.append(acc)
    loss_store_2.append(loss)

print("acc_1:",acc_1)
print("acc_2", acc_2)


x_axis = range(MAX)
#W = W - np.min(W)
#W = W/np.max(W)
pic = np.reshape(W[0,:],[32,32,3])
#dataNew = 'W.mat'
np.savetxt('W4.txt',W)

#pic = pic.transpose(2,1,3)
#pic2 = np.reshape(np.floor(data_1[:,1]*255),[32,32,3])
#pic2 = pic2.transpose(1,0,2)
plt.figure(1)
plt.imshow(pic)
#figure
plt.figure(2)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(x_axis,J_store_1,'r',label='training data')
plt.plot(x_axis,J_store_2,'g',label='validation data')
plt.legend()
plt.savefig('cost_4.jpg')
plt.figure(3)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(x_axis,acc_1,'r',label='training data')
plt.plot(x_axis,acc_2,'g',label='validation data')
plt.legend()
plt.savefig('accuracy_1.jpg')
plt.figure(4)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x_axis,loss_store_1,'r',label='training data')
plt.plot(x_axis,loss_store_2,'g',label='validation data')
plt.savefig('loss_4.jpg')
plt.legend()

plt.show()
