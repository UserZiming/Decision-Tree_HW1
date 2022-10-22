import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# implement the batch gradient descent
def stochasic_gradiant_decent(df, r, iterations):
    cost = np.empty(0)
    # y is the last column and x is all columns without the first column
    x = np.array(df.iloc[:, :-1])
    y = np.array(df.iloc[:, -1])

    # insert the bias term
    ones = np.ones(len(x))
    prime_x = np.insert(x, 0, ones, axis=1)

    # initialize weight w
    w = np.zeros(prime_x.shape[1])
    cost_f = np.empty(0)

    # start the iterations
    for i in range(iterations):
        for j in range(prime_x.shape[0]):
            predictions = np.dot(prime_x[j], w)
            cost = 1 / 2 * sum(np.square((y - np.dot(prime_x, w))))
            cost_f = np.append(cost_f, cost)
            grad = np.dot(prime_x[j], predictions - y[j])
            new_w = w - r * grad
            if (np.all(np.isclose(new_w, w, atol=1e-06))):
                w = new_w
                print('Stopped and converged after ', i, 'iterations')
                break
            w = new_w

    return w, cost_f


def test(row, w):
    x_in = np.array(row[0:-1])
    x_in_prime = np.insert(x_in, 0, 1)

    prediction = np.dot(w, x_in_prime)
    return prediction


# Gradient cost function
def cost_function(test_df, w):
    predictions = np.empty(0)
    for i in range(test_df.shape[0]):
        predictions = np.append(predictions, test(test_df.iloc[i, :], w))

    actual = np.array(test_df.iloc[:, -1])

    cost_test = 1 / 2 * sum(np.square(predictions - actual))

    print('The cost function(test dataset):', cost_test)
