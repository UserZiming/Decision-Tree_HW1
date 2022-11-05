import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def run(train_data, T, l):
    n_train = len(train_data)
    # create two numpy array for later computing
    train_data['bias'] = np.ones(n_train)
    w = np.zeros(len(train_data.iloc[0]) - 1)
    a = w

    # determine the number of epochs as a hyper-parameter
    for i in range(T):
        # for each training example belongs to training set
        for j in range(len(train_data)):
            x = train_data.drop(columns=['label']).iloc[j].to_numpy()
            y = train_data['label'].iloc[j]
            # compute the margin of the dataset
            margin = y * np.dot(w.T, x)
            # when the prediction is incorrect
            if margin <= 0:
                # update the weight
                w = w + l * y * x
            a = a + w
    return w, a


def predict(test_data, w, a):

    n_test = len(test_data)
    # create two numpy array for later computing
    test_data['bias'] = np.ones(len(test_data))
    predict = np.zeros(len(test_data))
    # iterate each test data and predict the label
    for i in range(n_test):
        x = test_data.iloc[i].to_numpy()
        # determine the label by using sign to compute the degree
        predict[i] = np.sign(np.dot(a.T, x))
    test_data['prediction'] = predict

    return test_data
