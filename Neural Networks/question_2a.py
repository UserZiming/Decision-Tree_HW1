import pandas as pd
import numpy as np
import neural_network


def sigma(num):
    return 1 / (1 + np.exp(-num))


# import train/test dataset from folder
train_data_csv = "bank-note/train.csv"
test_data_csv = "bank-note/test.csv"
# import the features manually
attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train_df = pd.read_csv(train_data_csv, names=attributes).astype(float)
test_df = pd.read_csv(test_data_csv, names=attributes).astype(float)
# convert the label value from 0 to -1
train_df.loc[train_df['label'] == 0, 'label'] = -1
test_df.loc[test_df['label'] == 0, 'label'] = -1

weights_1 = np.array([[-1, 1], [-2, 2], [-3, 3]])
weights_2 = np.array([[-1, 1], [-2, 2], [-3, 3]])
weights_3 = np.array([[-1], [2], [-1.5]])

# implement the back propagation algorithm
data = np.array([[1, 1]])
lr = 1.0
x = np.insert(data, [0], 1)
l_1 = sigma(x @ weights_1)
l_1 = np.insert(l_1, [0], 1)
l_2 = sigma(l_1 @ weights_2)
l_2 = np.insert(l_2, [0], 1)
y = (l_2 @ weights_3)[0]
dy = y - 1.0
short_w3 = np.repeat(weights_3[1:], 3, axis=1).T
short_z2 = np.repeat(np.expand_dims(l_2.T[1:], axis=1), 3, axis=1).T
full_z1 = np.repeat(np.expand_dims(l_1, axis=1), 3 - 1, axis=1)
short_z1 = np.repeat(np.expand_dims(l_1[1:], axis=1), 3, axis=1).T
mid_z2 = np.repeat(np.expand_dims(l_2.T[1:], axis=1), 3 - 1, axis=1)
data = np.repeat(np.expand_dims(x, axis=1), 3 - 1, axis=1)
middle_chunk = np.repeat(weights_3[1:], 3, axis=1).T @ (mid_z2 * (1 - mid_z2) * weights_2[1:].T)
weights_1 = weights_1 - lr * dy * middle_chunk * short_z1 * (1 - short_z1) * data
weights_2 = weights_2 - lr * dy * short_w3 * short_z2 * (1 - short_z2) * full_z1
weights_3 = weights_3 - lr * dy * np.expand_dims(l_2, axis=1)

print(l_1)
print(l_2)
print(weights_1)
print(weights_2)
print(weights_3)
