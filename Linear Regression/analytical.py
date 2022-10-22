import os
import sys
import pandas as pd
import numpy as np
import BatchGradientDescent
import matplotlib.pyplot as plt


def optimal_calculate(train_data, attributes):
    n_train = len(attributes)
    weight = np.zeros(n_train)
    y = train_data.T.iloc[-1].to_numpy()
    train_data['bias'] = np.ones(len(train_data))
    x = train_data.drop(columns=['label']).to_numpy()
    weight = np.linalg.inv(x.T @ x) @ x.T @ y.T

    error = 0
    for i in range(len(x)):
        error += (y[i] - weight.T @ x[i]) ** 2

    return weight
