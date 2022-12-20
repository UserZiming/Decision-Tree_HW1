import numpy as np
import pandas as pd
import random_tree


def run(train_data, test_data, attribute_types, T, sub_set_size):
    # set the train and test number
    sample_size = len(train_data)
    n_train = len(train_data)
    n_test = len(test_data)

    # initial the dataset for the later predict operations
    train_data['f_predict'] = np.zeros(n_train)
    test_data['f_predict'] = np.zeros(n_test)
    train_data['predict'] = np.zeros(n_train)
    test_data['predict'] = np.zeros(n_test)
    train_data['Miss'] = np.zeros(n_train)
    test_data['Miss'] = np.zeros(n_test)
    # record the error weight for each iteration
    train_error = np.zeros(T)

    # start the iteration to train weak classifiers
    for i in range(T):
        print("..... iteration ", i, ".....")
        sample = train_data.sample(sample_size, replace=True, ignore_index=True)
        sample = sample.drop(columns=['predict', 'Miss', 'f_predict'])
        # Set the depth as 100 since it will be stopped when out of stock
        tree = random_tree.RandomDecisionTree(sample, attribute_types, 100, sub_set_size)
        train_data = tree.testing(train_data)
        test_data = tree.testing(test_data)

        train_data['f_predict'] = train_data['f_predict'] + train_data['predict']
        test_data['f_predict'] = test_data['f_predict'] + test_data['predict']

        train_error[i] = sum(abs((train_data['label'] - np.sign(train_data['f_predict'])) / 2)) / len(train_data)

    prediction_train = np.sign(train_data['f_predict'].to_numpy())
    prediction = np.sign(test_data['f_predict'].to_numpy())

    return prediction_train, prediction
