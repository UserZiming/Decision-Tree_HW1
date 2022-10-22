import pandas as pd
import random
import numpy as np
import adaboost
import baggedTree
import decision_random_tree
import randomForest

import matplotlib.pyplot as plt

# import dataset from credit card folder
df_path = './credit card/default of credit card clients.csv'

# import the attributes and attributes types manually
attributes = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
              'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
              'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
              'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
              'default payment next month']
attributes_types = {
    'LIMIT_BAL': ['numeric'],
    'SEX': ['1', '2'],
    'EDUCATION': ['numeric'],
    'MARRIAGE': ['numeric'],
    'AGE': ['numeric'],
    'PAY_0': ['numeric'],
    'PAY_2': ['numeric'],
    'PAY_3': ['numeric'],
    'PAY_4': ['numeric'],
    'PAY_5': ['numeric'],
    'PAY_6': ['numeric'],
    'BILL_AMT1': ['numeric'],
    'BILL_AMT2': ['numeric'],
    'BILL_AMT3': ['numeric'],
    'BILL_AMT4': ['numeric'],
    'BILL_AMT5': ['numeric'],
    'BILL_AMT6': ['numeric'],
    'PAY_AMT1': ['numeric'],
    'PAY_AMT2': ['numeric'],
    'PAY_AMT3': ['numeric'],
    'PAY_AMT4': ['numeric'],
    'PAY_AMT5': ['numeric'],
    'PAY_AMT6': ['numeric'],
    'default payment next month': ['1', '0']}

# import dataset from the folder
df = pd.read_csv(df_path, names=attributes)

# Randomly choose 24000 train dataset and 6000 test dataset
df.sample(frac=1)
train_dataSet = df[:24000]
test_dataSet = df[24000:]

# Add several new columns for saving the results of calculate for each iteration
train_dataSet['f_prediction'] = np.zeros(len(train_dataSet))
test_dataSet['f_prediction'] = np.zeros(len(test_dataSet))
train_dataSet['prediction'] = np.zeros(len(train_dataSet))
test_dataSet['prediction'] = np.zeros(len(test_dataSet))
train_dataSet['miss'] = np.zeros(len(train_dataSet))
test_dataSet['miss'] = np.zeros(len(test_dataSet))

# Convert the "default payment next month" label to label and convert the yes or no to 1 or -1
train_dataSet = train_dataSet.rename(columns={'default payment next month': 'label'})
test_dataSet = test_dataSet.rename(columns={'default payment next month': 'label'})
train_dataSet['label'] = train_dataSet['label'].apply(lambda x: '1' if x == '1' else '-1').astype(float)
test_dataSet['label'] = test_dataSet['label'].apply(lambda x: '1' if x == '1' else '-1').astype(float)

# Convert the numerical features into binary by using the media of attribute values as the threshold
for a in attributes:
    if attributes_types[a][0] == 'numeric':
        median = train_dataSet[train_dataSet[a] != 'unknown'][a].astype(float).median()
        train_dataSet.loc[train_dataSet[a].astype(float) > median, a] = median + 1
        train_dataSet.loc[train_dataSet[a].astype(float) <= median, a] = median - 1
        test_dataSet.loc[test_dataSet[a].astype(float) > median, a] = median + 1
        test_dataSet.loc[test_dataSet[a].astype(float) <= median, a] = median - 1
        attributes_types[a] = [median + 1, median - 1]


res = input("Please input the different number: (1. Adaboost 2. Bagged Tree 3. Random Forest)")

if res == '1':
    # Set iteration times for adaboost algorithm
    T = 500
    print("We totally have ", T, " iterations of AdaBoost Algorithm")
    adaboost_out = adaboost.adaboost(train_dataSet, test_dataSet, attributes_types, T)
    print(adaboost_out)

    # Display the second figure for the results
    x = range(1, T + 1)
    fig1 = plt.figure(1)
    ax1 = plt.axes()
    ax1.plot(x, adaboost_out[0], c='b', label='Train Accuracy')
    ax1.plot(x, adaboost_out[1], c='r', label='Test Accuracy')
    ax1.set_title("Adaboost")
    plt.legend()
    plt.show()
elif res == '2':

    # Set iteration times for begged tree algorithm
    T = 500
    print("We totally have ", T, " iterations of bagged tree")
    bagged_out = baggedTree.bagged_tree(train_dataSet, test_dataSet, attributes_types, T)
    print(bagged_out)

    # Display the second figure for the results
    x = range(1, T + 1)
    fig2 = plt.figure(2)
    ax2 = plt.axes()
    ax2.plot(x, bagged_out[0], c='b', label='Train Accuracy')
    ax2.plot(x, bagged_out[1], c='r', label='Test Accuracy')
    ax2.set_title("Bagged Tree")
    plt.legend()
    plt.show()

elif res == '3':

    # Set iteration times for random forest algorithm
    T = 500
    sub_set_size = 2
    # sub_set_size = 4
    # sub_set_size = 6
    print("We totally have ", T, " iterations of random forest with ", sub_set_size, " subset size")

    random_forest_out = randomForest.rand_forest(train_dataSet, test_dataSet, attributes_types, T, sub_set_size)
    print(random_forest_out)

    # Display the second figure for the results
    x = range(1, T + 1)
    fig3 = plt.figure(2)
    ax3 = plt.axes()
    ax3.plot(x, random_forest_out[0], c='b', label='Train Accuracy')
    ax3.plot(x, random_forest_out[1], c='r', label='Test Accuracy')
    ax3.set_title("Random Forest")
    plt.legend()
    plt.show()

else:
    print("Wrong number, default to run Begged Tree")
    # Set iteration times for begged tree algorithm
    T = 500
    print("We totally have ", T, " iterations of bagged tree")
    bagged_out = baggedTree.bagged_tree(train_dataSet, test_dataSet, attributes_types, T)
    print(bagged_out)

    # Display the second figure for the results
    x = range(1, T + 1)
    fig2 = plt.figure(2)
    ax2 = plt.axes()
    ax2.plot(x, bagged_out[0], c='b', label='Train Accuracy')
    ax2.plot(x, bagged_out[1], c='r', label='Test Accuracy')
    ax2.set_title("Bagged Tree")
    plt.legend()
    plt.show()
