import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


class neural_network:
    def __init__(self, l_size, d):
        self.layer_size = l_size
        self.dim = d
        self.x = np.zeros((1, d + 1))
        self.y = 0
        # initialize parameters weights
        self.w_1 = np.random.normal(0, 0.5, (d + 1, l_size - 1))
        self.w_2 = np.random.normal(0, 0.5, (l_size, l_size - 1))
        self.w_3 = np.random.normal(0, 0.5, (l_size, 1))
        # initialize the layers
        self.l_1 = np.zeros((1, l_size - 1))
        self.l_2 = np.zeros((1, l_size - 1))

    def sigma_function(self, num):
        return 1 / (1 + np.exp(-num))

    def run(self, train_df, lr_0, d, epochs):
        t = 1
        n_train = len(train_df)
        # for each epoch
        for T in range(epochs):
            # shuffle the training set
            shuffled_data = train_df.sample(n_train, replace=False, ignore_index=True)
            # iterate each training set
            for j in range(len(shuffled_data)):
                t += 1
                # apply the schedule of learning rate
                lr = lr_0 / (1 + (lr_0 / d) * t)
                x = shuffled_data.drop(columns=['label']).iloc[j].to_numpy()
                y = shuffled_data['label'].iloc[j]
                # implement the back propagation algorithm
                self.x = np.insert(x, [0], 1)
                # self.l_1 = self.sigma_function(self.x @ self.w_1)
                self.l_1 = np.insert(self.sigma_function(self.x @ self.w_1), [0], 1)
                # self.l_2 = self.sigma_function(self.l_1 @ self.w_2)
                self.l_2 = np.insert(self.sigma_function(self.l_1 @ self.w_2), [0], 1)
                self.y = (self.l_2 @ self.w_3)[0]
                dy = self.y - y
                data = np.repeat(np.expand_dims(self.x, axis=1), self.layer_size - 1, axis=1)
                full_z1 = np.repeat(np.expand_dims(self.l_1, axis=1), self.layer_size - 1, axis=1)
                short_z1 = np.repeat(np.expand_dims(self.l_1[1:], axis=1), self.dim + 1, axis=1).T
                short_z2 = np.repeat(np.expand_dims(self.l_2.T[1:], axis=1), self.layer_size, axis=1).T
                mid_z2 = np.repeat(np.expand_dims(self.l_2.T[1:], axis=1), self.layer_size - 1, axis=1)
                short_w3 = np.repeat(self.w_3[1:], self.layer_size, axis=1).T
                # update the by using the gradient of the loss
                self.w_1 = self.w_1 - lr * dy * (np.repeat(self.w_3[1:], self.dim + 1, axis=1).T @ (
                        mid_z2 * (1 - mid_z2) * self.w_2[1:].T)) * short_z1 * (1 - short_z1) * data
                self.w_2 = self.w_2 - lr * dy * short_w3 * short_z2 * (1 - short_z2) * full_z1
                self.w_3 = self.w_3 - lr * dy * np.expand_dims(self.l_2, axis=1)

    def test(self, test_df):
        n_test = len(test_df)
        n_test_df = np.zeros(n_test)
        for i in range(n_test):
            self.x = np.insert(test_df.iloc[i].to_numpy(), [0], 1)
            # self.l_1 = self.sigma_function(self.x @ self.w_1)
            self.l_1 = np.insert(self.sigma_function(self.x @ self.w_1), [0], 1)
            # self.l_2 = self.sigma_function(self.l_1 @ self.w_2)
            self.l_2 = np.insert(self.sigma_function(self.l_1 @ self.w_2), [0], 1)
            self.y = (self.l_2 @ self.w_3)[0]
            n_test_df[i] = self.y

        return n_test_df
