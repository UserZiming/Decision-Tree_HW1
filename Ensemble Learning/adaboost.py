import numpy as np
import pandas as pd
import decision_tree
import matplotlib.pyplot as plt


# implement adaboost algorithm
def adaboost(train_data, test_data, attribute_types, T):

    # set the train and test number
    n_train = len(train_data)
    n_test = len(test_data)

    # set the initial weight for each data
    w = np.ones(n_train)
    w = w / len(train_data)
    train_data['probability'] = w

    # record the error weight for each iteration and weak classifier
    train_error = np.zeros(T)
    test_error = np.zeros(T)
    train_tree_error = np.zeros(T)
    test_tree_error = np.zeros(T)

    # start the iteration to train weak classifiers
    for i in range(T):
        print("..... iteration ", i, ".....")
        s = train_data.sample(n_train, replace=True, weights=train_data['probability'], ignore_index=True).drop(
            columns=['probability', 'f_prediction', 'prediction', 'miss'])
        # train the dataset by using one depth of decision tree
        tree = decision_tree.DecisionTree(s, attribute_types, 1)
        train_data = tree.testing(train_data)
        test_data = tree.testing(test_data)

        # assign the error into the dataset
        train_tree_error[i] = sum(train_data['miss']) / len(train_data['miss'])
        test_tree_error[i] = sum(test_data['miss']) / len(test_data['miss'])

        # calculate error for each weight, Then according to the algorithm,
        # the weight of the wrong point will be increased
        w_error = sum(train_data['probability'] * train_data['miss'])
        alpha = 0.5 * np.log((1 - w_error) / w_error)
        # add to prediction to de dataset
        train_data['f_prediction'] = train_data['f_prediction'] + alpha * train_data['prediction']
        test_data['f_prediction'] = test_data['f_prediction'] + alpha * test_data['prediction']
        train_data['probability'] = train_data['probability'] * np.exp(
            -alpha * train_data['label'] * train_data['prediction'])
        train_data['probability'] = train_data['probability'] / sum(train_data['probability'])
        train_error[i] = sum(abs((train_data['label'] - np.sign(train_data['f_prediction'])) / 2)) / n_train
        test_error[i] = sum(abs((test_data['label'] - np.sign(test_data['f_prediction'])) / 2)) / n_test

    # return final results
    return train_tree_error, test_tree_error, train_error, test_error
