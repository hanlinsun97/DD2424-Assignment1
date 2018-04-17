import numpy as np
import matplotlib.pyplot as plt

from LoadBatch_1 import LoadBatch_1
from LoadBatch_2 import LoadBatch_2
from initialization import initialization

import scipy.io as sio

#Parameter
batch_size = 100
lam = 0.01
MAX = 40
learning_rate = 0.05
training_data = 10000

# define some functions
def svm_loss_naive(W, X, y, reg):
    W = np.transpose(W)
    X = np.transpose(X)
    dW = np.zeros(W.shape) # initialize the gradient as zero
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    h = 0.00001
    hW = W - h
    for i in range(num_train):
        scores = X[i].dot(W)
        hscores = X[i].dot(hW)
        
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] += -X[i]
                dW[:, j] += X[i]

    loss /= num_train
    dW /= num_train
    cost = loss + reg * np.sum(W * W)
    dW += 2 * reg * W         
    dW = np.transpose(dW)
    scores = np.transpose(scores)
    return loss,cost, dW,scores

def svm_loss_vectorized(W, X, y, reg):
    W = np.transpose(W)
    X = np.transpose(X)

    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    scores = X.dot(W)  # N by C
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores_correct = scores[np.arange(num_train), y] #1 by N
    scores_correct = np.reshape(scores_correct, (num_train, 1)) # N by 1
    margins = scores - scores_correct + 1.0 # N by C
    margins[np.arange(num_train), y] = 0.0 
    margins[margins <= 0] = 0.0
    loss += np.sum(margins) / num_train
    cost = loss + reg * np.sum(W * W)
    margins[margins > 0] = 1.0                         
    row_sum = np.sum(margins, axis=1)                  # 1 by N
    margins[np.arange(num_train), y] = -row_sum        
    dW += np.dot(X.T, margins)/num_train + 2*reg * W     # D by C
    dW = np.transpose(dW)
    scores = np.transpose(scores)
    return cost,loss, dW,scores


def ComputeAccuracy(scores, input_label_no_onehot, batch_size):
    Q = scores.argmax(axis=0)  # Predict label
    #print(Q)
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
    learning_rate = learning_rate * 0.9# Decay
    for i in range(int(training_data/batch_size)):

        input_data = data_1[:,i*batch_size:(i+1)*batch_size]
        input_label = label_1[:,i*batch_size:(i+1)*batch_size]
        input_label_no_onehot = label_no_onehot_1[i*batch_size:(i+1)*batch_size]
        loss,cost, dW,scores = svm_loss_vectorized(W, input_data, input_label_no_onehot, lam)
        W = W - learning_rate*dW
        

    loss,cost, dW,scores = svm_loss_vectorized(W, data_1, label_no_onehot_1, lam)
    acc = ComputeAccuracy(scores,label_no_onehot_1, data_length_1)
    J_store_1.append(cost)
    acc_1.append(acc)
    loss_store_1.append(loss)
# We run our model on validation set

    loss,cost, dW,scores = svm_loss_vectorized(W, data_2, label_no_onehot_2, lam)
    acc = ComputeAccuracy(scores,label_no_onehot_2, data_length_2)
    J_store_2.append(cost)
    acc_2.append(acc)
    loss_store_2.append(loss)
print("acc_1:",acc_1)
print("acc_2", acc_2)

x_axis = range(MAX)

np.savetxt('W_svm.txt',W)

#figure
plt.figure(2)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(x_axis,J_store_1,'r',label='training data')
plt.plot(x_axis,J_store_2,'g',label='validation data')
plt.legend()
plt.savefig('cost_svm.png')
plt.figure(3)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(x_axis,acc_1,'r',label='training data')
plt.plot(x_axis,acc_2,'g',label='validation data')
plt.legend()
#plt.savefig('accuracy_1.jpg')
plt.figure(4)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x_axis,loss_store_1,'r',label='training data')
plt.plot(x_axis,loss_store_2,'g',label='validation data')
plt.savefig('loss_svm.png')
plt.legend()

plt.show()
