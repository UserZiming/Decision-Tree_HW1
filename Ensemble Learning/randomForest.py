import pandas as pd
import numpy as np
import math
import decision_random_tree
import matplotlib.pyplot as plt


def rand_forest(train_data, test_data, attribute_types, T, sub_size):
    # set the train and test number
    n_train = len(train_data)
    n_test = len(test_data)

    # record the error weight for each iteration
    train_error = np.zeros(T)
    test_error = np.zeros(T)

    # record the first trained data for computing
    first_test_data = []

    # start the iteration to train weak classifiers
    for i in range(T):
        print("..... iteration ", i, ".....")
        s = train_data.sample(n_train, replace=True, ignore_index=True).drop(
            columns=['f_prediction', 'prediction', 'miss'])
        # Set the depth as 100 since it will be stopped when out of stock
        tree = decision_random_tree.DecisionTree(s, attribute_types, 100, sub_size)
        train_data = tree.testing(train_data)
        test_data = tree.testing(test_data)

        train_data['f_prediction'] = train_data['f_prediction'] + train_data['prediction']
        test_data['f_prediction'] = test_data['f_prediction'] + test_data['prediction']
        train_error[i] = sum(abs((train_data['label']
                                  - np.sign(train_data['f_prediction'])) / 2)) / n_train
        test_error[i] = sum(abs((test_data['label']
                                 - np.sign(test_data['f_prediction'])) / 2)) / n_test

        # store the first train data
        if i == 0:
            first_test_data = test_data.copy()

    return train_error, test_error, test_data, first_test_data
