import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import primal_SVM


def print_prediction(train_df, test_df, w):
    prediction_train = primal_SVM.predict(train_df.drop(columns=['label']), w)
    prediction_test = primal_SVM.predict(test_df.drop(columns=['label']), w)

    train_error = sum(abs(prediction_train['prediction'] - train_df['label'])) / (2 * len(train_df))
    test_error = sum(abs(prediction_test['prediction'] - test_df['label'])) / (2 * len(test_df))

    print('Train Error: ' + str(train_error))
    print('Test Error: ' + str(test_error))


attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
weights = []

# import training and testing dataset
train_data_path = "classification/train.csv"
test_data_path = "classification/test.csv"
train_df = pd.read_csv(train_data_path, names=attributes).astype(float)
test_df = pd.read_csv(test_data_path, names=attributes).astype(float)
train_df.loc[train_df['label'] == 0, 'label'] = -1
test_df.loc[test_df['label'] == 0, 'label'] = -1

# Tuning parameters
T = 100
C_list = [100/873, 500/873, 700/873]
learning_rate = 0.001
a = 0.001

# run the primal SVM with a
for C in C_list:
    print('----------Using learning rate with a and C is', str(C), '------------')
    w = primal_SVM.run(train_df, C, T, learning_rate, a, 'a')
    weights.append(w)
    print_prediction(train_df, test_df, w)

# run the primal SVM without a
for C in C_list:
    print('----------Using learning rate without a: and C is', str(C), '-------------')
    w = primal_SVM.run(train_df, C, T, learning_rate, a, 'not a')
    weights.append(w)
    print_prediction(train_df, test_df, w)
