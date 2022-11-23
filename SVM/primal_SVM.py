import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def schedule_learning_a(t, a, learning_rate):
    return learning_rate / (1 + (learning_rate * t) / a)


def schedule_learning_no_a(t, learning_rate):
    return learning_rate / (1 + t)


def run(train_data, C, T, learning_rate, a, learn_type):
    n_train = len(train_data)
    # create a column for saving
    train_data['bias'] = np.ones(n_train)
    # initialize the weight
    w = np.zeros(len(train_data.iloc[0]) - 1)

    if learn_type == 'a':
        for i in range(T):
            # use the schedule of learning rate with a
            learn_r = schedule_learning_a(i, a, learning_rate)
            # shuffle the training dataset
            data = train_data.sample(n_train, replace=False, ignore_index=True)
            # iterate each training example
            for j in range(len(data)):
                x = data.drop(columns=['label']).iloc[j].to_numpy()
                y = data['label'].iloc[j]
                if y * np.dot(w.T, x) <= 1:
                    # update the weight
                    w = w - learning_rate * np.append(w[:len(w) - 1], 0) \
                        + learning_rate * C * n_train * y * x
                else:
                    # update the initial weight
                    w[:len(w) - 1] = (1 - learn_r) * w[:len(w) - 1]
    else:
        for i in range(T):
            # use the schedule of learning rate without a
            learn_r = schedule_learning_no_a(i, learning_rate)
            # shuffle the training dataset
            data = train_data.sample(n_train, replace=False, ignore_index=True)
            # iterate each training example
            for j in range(len(data)):
                x = data.drop(columns=['label']).iloc[j].to_numpy()
                y = data['label'].iloc[j]
                if y * np.dot(w.T, x) <= 1:
                    # update the weight
                    w = w - learning_rate * np.append(w[:len(w) - 1], 0) \
                        + learning_rate * C * n_train * y * x
                else:
                    # update the initial weight
                    w[:len(w) - 1] = (1 - learn_r) * w[:len(w) - 1]

    return w


def predict(test_data, w):
    n_test = len(test_data)
    prediction = np.zeros(len(test_data))
    # create a column for saving
    test_data['bias'] = np.ones(len(test_data))
    # iterate each test data and predict the label
    for i in range(n_test):
        x = test_data.iloc[i].to_numpy()
        # determine the label by using sign to compute the degree
        prediction[i] = np.sign(np.dot(w.T, x))
    # save the result into prediction column
    test_data['prediction'] = prediction
    return test_data
