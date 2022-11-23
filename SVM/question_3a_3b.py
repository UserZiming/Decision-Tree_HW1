import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import dual_SVM


# def print_prediction(train_df, test_df, x, y, a, b, theta, kernel_type):
#     prediction_train = dual_SVM.predict(train_df.drop(columns=['label']), x, y, a, b, theta, kernel_type)
#     prediction_test = dual_SVM.predict(test_df.drop(columns=['label']), x, y, a, b, theta, kernel_type)
#     print('C is' + str(C))
#     print('Train Error: ' + str(sum(abs(prediction_train['prediction'] - train_df['label'])) / (2 * len(train_df))))
#     print('Test Error: ' + str(sum(abs(prediction_test['prediction'] - test_df['label'])) / (2 * len(test_df))))


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
C_list = [100 / 873, 500 / 873, 700 / 873]
theta_list = [0.1, 0.5, 1, 5, 100]
a = 0.001

# C_list = [500 / 873]
# theta_list = [0.1, 0.5]

# run the dual SVM with linear function
for C in C_list:
    print('------------Run the dual SVM with linear function-------------')
    x, y, a, b, w = dual_SVM.run(train_df, C, 0.01, 0)
    weights.append(np.append(w, b))
    prediction_train = dual_SVM.predict(train_df.drop(columns=['label']), x, y, a, b, 0.01, 0)
    prediction_test = dual_SVM.predict(test_df.drop(columns=['label']), x, y, a, b, 0.01, 0)
    print('C is' + str(C))
    print('Train Error: ' + str(sum(abs(prediction_train['prediction'] - train_df['label'])) / (2 * len(train_df))))
    print('Test Error: ' + str(sum(abs(prediction_test['prediction'] - test_df['label'])) / (2 * len(test_df))))

# run the dual SVM with Gaussian kernel function
for theta in theta_list:
    for C in C_list:
        print('------------run the dual SVM with Gaussian kernel function and the theta is ', str(theta),
              '------------')
        x, y, a, b, w = dual_SVM.run(train_df, C, theta, 1)
        prediction_train = dual_SVM.predict(train_df.drop(columns=['label']), x, y, a, b, theta, 1)
        prediction_test = dual_SVM.predict(test_df.drop(columns=['label']), x, y, a, b, theta, 1)
        print('C is' + str(C))
        print('Train Error: ' + str(sum(abs(prediction_train['prediction'] - train_df['label'])) / (2 * len(train_df))))
        print('Test Error: ' + str(sum(abs(prediction_test['prediction'] - test_df['label'])) / (2 * len(test_df))))
