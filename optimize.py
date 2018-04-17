# Use 5 data sets

import numpy as np
import matplotlib.pyplot as plt
from LoadBatch_1 import LoadBatch_1
from LoadBatch_2 import LoadBatch_2
from LoadBatch_3 import LoadBatch_3
from LoadBatch_4 import LoadBatch_4
from LoadBatch_5 import LoadBatch_5
from initialization import initialization

import scipy.io as sio



#Parameter
batch_size = 100
lam = 0.01
MAX = 40
learning_rate = 0.05
training_data = 49000

# define some functions
def ComputeCost(label, W, lam, batch_size, P):
    Y = label
    loss = -(1.0 / batch_size) * np.sum(Y * np.log(P))
    J = loss + lam * np.sum(np.power(W, 2))
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

    grad_W = np.dot((P - input_label), np.transpose(input_data))/batch_size+2*lam*W
    grad_b = np.mean(P-input_label, 1)
    grad_b = np.reshape(grad_b, [10, 1])
    return grad_W, grad_b

def ComputeAccuracy(P, input_label_no_onehot, batch_size):
    Q = P.argmax(axis=0)  # Predict label
    #print(Q)
    Y = input_label_no_onehot
    diff = Q - Y
    acc = np.sum(diff == 0)/batch_size
    return acc


#Load data and initialization

[data_1, label_1, data_length_1, label_no_onehot_1] = LoadBatch_1()
[data_2, label_2, data_length_2, label_no_onehot_2] = LoadBatch_2()
[data_3, label_3, data_length_3, label_no_onehot_3] = LoadBatch_3()
[data_4, label_4, data_length_4, label_no_onehot_4] = LoadBatch_4()
[data_5, label_5, data_length_5, label_no_onehot_5] = LoadBatch_5()

data_real = np.concatenate((data_1,data_2,data_3,data_4,data_5[:,0:9000]),axis=1)

label_real = np.concatenate((label_1,label_2,label_3,label_4,label_5[:,0:9000]),axis=1)


data_valid = data_5[:,9000:10000]
label_valid = label_5[:,9000:10000]

label_no_onehot_real=np.concatenate((label_no_onehot_1,label_no_onehot_2,label_no_onehot_3,label_no_onehot_4,label_no_onehot_5[0:9000]),axis=0)
#label_no_onehot_real = np.reshape(label_no_onehot_real,[1,49000])

label_no_onehot_valid=label_no_onehot_5[9000:10000]

data_length_real = 49000
data_length_valid = 1000

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
    learning_rate = 0.85 * learning_rate
  #  print(data_real.shape)
  #  print(label_real.shape)
  #  print(label_no_onehot_real.shape)
  #  print(data_valid.shape)
  #  print(label_valid.shape)
  #  print(label_no_onehot_valid.shape)


    for i in range(int(training_data/batch_size)):

        input_data = data_real[:,i*batch_size:(i+1)*batch_size]
        input_label = label_real[:,i*batch_size:(i+1)*batch_size]
        input_label_no_onehot = label_no_onehot_real[i*batch_size:(i+1)*batch_size]
        P = EvaluationClassifier(W, b, input_data)  # 10 * Batch_size
        grad_W, grad_b = ComputeGradients(W,b,P,input_data,input_label,lam,batch_size)
        W = W - learning_rate*grad_W
        b = b - learning_rate*grad_b

    P_use = EvaluationClassifier(W, b, data_real)
    J,loss = ComputeCost(label_real, W, lam, data_length_real, P_use)
    acc = ComputeAccuracy(P_use,label_no_onehot_real, data_length_real)
    J_store_1.append(J)
    acc_1.append(acc)
    loss_store_1.append(loss)
 #   print(acc_1)
# We run our model on validation set


    P_use = EvaluationClassifier(W, b, data_valid)
    J,loss = ComputeCost(label_valid, W, lam, data_length_valid, P_use)
    acc = ComputeAccuracy(P_use, label_no_onehot_valid, data_length_valid)
    J_store_2.append(J)
    acc_2.append(acc)
    loss_store_2.append(loss)

print("acc_1:",acc_1)
print("acc_2", acc_2)



x_axis = range(MAX)
#W = W - np.min(W)
#W = W/np.max(W)

#dataNew = 'W.mat'
np.savetxt('W_opti.txt',W)

#pic = pic.transpose(2,1,3)
#pic2 = np.reshape(np.floor(data_1[:,1]*255),[32,32,3])
#pic2 = pic2.transpose(1,0,2)
#plt.figure(1)
#plt.imshow(pic)
#figure
plt.figure(2)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(x_axis,J_store_1,'r',label='training data')
plt.plot(x_axis,J_store_2,'g',label='validation data')
plt.legend()
plt.savefig('cost_opti.jpg')
plt.figure(3)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(x_axis,acc_1,'r',label='training data')
plt.plot(x_axis,acc_2,'g',label="validation data")
plt.legend()
#plt.savefig('accuracy_1.jpg')
plt.figure(4)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x_axis,loss_store_1,'r',label='training data')
plt.plot(x_axis,loss_store_2,'g',label="validation data")
plt.savefig('loss_opti.jpg')
plt.legend()

plt.show()
