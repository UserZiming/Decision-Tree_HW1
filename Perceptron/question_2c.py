import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import average_perceptron


# import train/test dataset from folder
train_data_csv = "bank-note/train.csv"
test_data_csv = "bank-note/test.csv"

# import the features manually
attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
train_df = pd.read_csv(train_data_csv, names=attributes).astype(float)
test_df = pd.read_csv(test_data_csv, names=attributes).astype(float)
# convert the label value from 0 to -1
train_df.loc[train_df['label'] == 0, 'label'] = -1
test_df.loc[test_df['label'] == 0, 'label'] = -1

# drop the correct label from test dataset
test_df_nolabel = test_df.drop(columns=['label'])

# hyper-parameters
lr = 0.000001
T = 10

# run average perceptron and get predictions on test dataset
w, a = average_perceptron.run(train_df, T, lr)
after_test_df = average_perceptron.predict(test_df_nolabel, w, a)

# compute average prediction error
average_prediction_error = sum(abs(after_test_df['prediction'] - test_df['label'])) / 2 / len(test_df)
print('Learned average weight vectors: ', a)
print('Average prediction error: ' + str(average_prediction_error))
