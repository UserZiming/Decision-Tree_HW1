import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def run(train_data, T, l):
    n_train = len(train_data)
    weight_list = []
    # c is the number of predictions made by w - correct counts!
    c = 0
    c_list = []
    # create two numpy array for later computing
    train_data['bias'] = np.ones(n_train)
    w = np.zeros(len(train_data.iloc[0]) - 1)

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
                # add c and w into lists
                c_list.append(c)
                weight_list.append(w)
                # update the weight
                w = w + l * y * x
                c = 1
            else:
                c = c + 1
        # add c and w into lists
        c_list.append(c)
        weight_list.append(w)
    return weight_list, c_list


def predict(test_data, weight_list, c_list):
    n_test = len(test_data)
    n_weight = len(weight_list)
    # create two numpy array for later computing
    test_data['bias'] = np.ones(len(test_data))
    predict = np.zeros(len(test_data))
    # iterate each test data and predict the label
    for i in range(n_test):
        x = test_data.iloc[i].to_numpy()
        sum = 0
        for j in range(n_weight):
            sum = sum + c_list[j] * np.sign(np.dot(weight_list[j].T, x))
        # determine the label by using sign to compute the degree
        predict[i] = np.sign(sum)

    test_data['prediction'] = predict
    return test_data
