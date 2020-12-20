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

# initializing the W and b
weights = np.random.normal(0, 1, 3).reshape(3, 1)
n_epoch = 100
lr = 1

# training
n = len(train_set)
inputs = np.append(train_set.values[:, 0:2], np.ones(n).reshape(n, 1), 1)
yt = train_set.values[:, 2:3].reshape(n, 1)
for i in range(0, n_epoch):
    y = 1 / (1 + np.exp(-inputs.dot(weights)))
    grad = inputs.transpose().dot(y - yt)
    weights -= lr * grad / n

# evaluate the prediction
true_predicts = 0
for sample in test_set.values:
    if sample[0] * weights[0] + sample[1] * weights[1] + weights[2] > 0:
        if sample[2] == 1:
            true_predicts += 1
        plt.scatter(sample[0], sample[1], color='r')
    else:
        if sample[2] == 0:
            true_predicts += 1
        plt.scatter(sample[0], sample[1], color='b')
print("Accuracy: {}".format(true_predicts/len(test_set)))
plt.show()


