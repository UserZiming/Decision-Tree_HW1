import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Linear separable function
def dot(x1, x2):
    return x1@x2.T


# Gaussian kernel function
def gaussian(x1, x2, theta):
    return np.exp((- (np.tile(x1.T[0], (len(x1), 1)) - np.tile(x2.T[0], (len(x1), 1)).T) ** 2
                   - (np.tile(x1.T[1], (len(x1), 1)) - np.tile(x2.T[1], (len(x1), 1)).T) ** 2
                   - (np.tile(x1.T[2], (len(x1), 1)) - np.tile(x2.T[2], (len(x1), 1)).T) ** 2
                   - (np.tile(x1.T[3], (len(x1), 1)) - np.tile(x2.T[3], (len(x1), 1)).T) ** 2) / theta)


# dual form SVM
def run(train_data, C, theta, kernel_type):
    n_train = len(train_data)
    x = train_data.drop(columns=['label']).to_numpy()
    y = np.expand_dims(train_data['label'].to_numpy(), axis=0).T
    w = np.zeros(len(x))
    a = np.zeros(n_train)
    b = 0

    # when kernel type is 0, then we run the dual SVM with linear function
    if kernel_type == 0:
        # Define a function for the minimize function from scipy.optimize.minimize
        def fun(a):
            out = -(sum(a) - 0.5 * sum(sum(np.expand_dims(a, axis=0) @ np.expand_dims(a, axis=0).T
                                              * y@y.T * dot(x, x))))
            return out
        # define the constraints for minimize function
        cons = [{'type': 'eq', 'fun': lambda a: a.T@y},
                {'type': 'ineq', 'fun': lambda a: C * np.ones(len(a)) - a}]
        # the optimization result represented as a OptimizeResult object.
        res = minimize(fun, a, method='SLSQP', constraints=cons)
        a = res.x
        w = sum(np.expand_dims(a, axis=0).T * y * x)
        b = sum(np.expand_dims(a, axis=0).T * y)
    # when kernel type is 1, then we run the dual SVM with Gaussian kernel function
    else:
        # define a function for the minimize function from scipy.optimize.minimize
        def fun(a):
            out = -(sum(a) - 0.5 * sum(sum(np.expand_dims(a, axis=0) @ np.expand_dims(a, axis=0).T
                                           * y@y.T * gaussian(x, x, theta))))
            return out

        # define the constraints for minimize function
        cons = [{'type': 'eq', 'fun': lambda a: a.T@y},
                {'type': 'ineq', 'fun': lambda a: C * np.ones(len(a)) - a}]
        # the optimization result represented as a OptimizeResult object.
        res = minimize(fun, a, method='SLSQP', constraints=cons)
        a = res.x
        w = sum(np.expand_dims(a, axis=0).T * y * x)
        b = sum(np.expand_dims(a, axis=0).T * y)

    return x, y, a, b, w


def predict(test_data, x, y, a, b, theta, kernel_type):
    # when kernel type is 0, then we predict the data in linear function
    n_test = len(test_data)
    if kernel_type == 0:
        prediction = np.zeros(n_test)
        # iterate each test data and predict the label
        for i in range(n_test):
            x_i = test_data.iloc[i].to_numpy()
            kernel_x = dot(x, x_i)
            holder = sum(sum(np.expand_dims(a, axis=0) * y.T * kernel_x)) + b
            # determine the label by using sign to compute the degree
            prediction[i] = np.sign(holder)
        # save the result into prediction column
        test_data['prediction'] = prediction
    # when kernel type is 0, then we predict the data in Gaussian kernel function
    else:
        prediction = np.zeros(n_test)
        # iterate each test data and predict the label
        for i in range(n_test):
            x_i = test_data.iloc[i].to_numpy()
            kernel_x = gaussian(x, x_i, theta)
            holder = sum(sum(np.expand_dims(a, axis=0) * y.T * kernel_x)) + b
            # determine the label by using sign to compute the degree
            prediction[i] = np.sign(holder)
        # save the result into prediction column
        test_data['prediction'] = prediction

    return test_data
