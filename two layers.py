import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading the dataset
data = pd.read_csv("dataset.csv")
data_label0 = data[data['Label'] == 0]
data_label1 = data[data['Label'] == 1]

# splitting the dataset into 2 sets: train set and test set
train_set = data.sample(frac=0.8)
test_set = data.drop(train_set.index)

# plotting the dataset
# plt.scatter(data_label0['X1'], data_label0['X2'], color='r')
# plt.scatter(data_label1['X1'], data_label1['X2'], color='b')
# plt.show()

# initializing the weights
W_weights = np.random.normal(0, 1, 3).reshape(3, 1)
V_weights = np.random.normal(0, 1, 3).reshape(3, 1)
U_weights = np.random.normal(0, 1, 3).reshape(3, 1)
n_epoch = 10000
lr = 10

# training
n = len(train_set)
inputs = np.append(train_set.values[:, 0:2], np.ones(n).reshape(n, 1), 1)
yt = train_set.values[:, 2:3].reshape(n, 1)
for i in range(0, n_epoch):
    z0 = 1 / (1 + np.exp(-inputs.dot(W_weights)))
    z1 = 1 / (1 + np.exp(-inputs.dot(V_weights)))
    z = np.append(np.append(z0, z1, 1), np.ones(n).reshape(n, 1), 1)
    y = 1 / (1 + np.exp(-z.dot(U_weights)))

    W_grad = inputs.transpose().dot(2 * (y - yt) * U_weights[0] * y * (1 - y) * z0 * (1 - z0))
    V_grad = inputs.transpose().dot(2 * (y - yt) * U_weights[1] * y * (1 - y) * z1 * (1 - z1))
    U_grad = z.transpose().dot(2 * (y - yt) * y * (1 - y))
    W_weights -= lr * W_grad / n
    V_weights -= lr * V_grad / n
    U_weights -= lr * U_grad / n

# evaluate the prediction
true_predicts = 0
n = len(test_set)
test_inputs = np.append(test_set.values[:, 0:2], np.ones(n).reshape(n, 1), 1)
z0 = 1 / (1 + np.exp(-test_inputs.dot(W_weights)))
z1 = 1 / (1 + np.exp(-test_inputs.dot(V_weights)))
z = np.append(np.append(z0, z1, 1), np.ones(n).reshape(n, 1), 1)
output = z.dot(U_weights)
for i in range(n):
    if output[i] > 0:
        if test_set.values[i, 2] == 1:
            true_predicts += 1
        plt.scatter(test_set.values[i, 0], test_set.values[i, 1], color='r')
    else:
        if test_set.values[i, 2] == 0:
            true_predicts += 1
        plt.scatter(test_set.values[i, 0], test_set.values[i, 1], color='b')
print("Accuracy: {}".format(true_predicts/n))
plt.show()


