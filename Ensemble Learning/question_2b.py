import pandas as pd
import numpy as np
import math
import decision_tree
import matplotlib.pyplot as plt
import baggedTree


def Question_2b():
    # import dataset from bank folder
    train_dataSet = "./bank/train.csv"
    test_dataSet = "./bank/test.csv"

    # import the attributes and attributes types manually
    attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                  'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    attributes_types = {
        'age': ['numeric'],
        'job': ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                "blue-collar", "self-employed", "retired", "technician", "services"],
        'marital': ["married", "divorced", "single"],
        'education': ["unknown", "secondary", "primary", "tertiary"],
        'default': ['yes', 'no'],
        'balance': ['numeric'],
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ["unknown", "telephone", "cellular"],
        'day': ['numeric'],
        'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        'duration': ['numeric'],
        'campaign': ['numeric'],
        'pdays': ['numeric'],
        'previous': ['numeric'],
        'poutcome': ["unknown", "other", "failure", "success"],
        'y': ['yes', 'no']}

    # import dataset from the folder
    train_dataSet = pd.read_csv(train_dataSet, names=attributes)
    test_dataSet = pd.read_csv(test_dataSet, names=attributes)

    # Add several new columns for saving the results of calculate for each iteration
    train_dataSet['f_prediction'] = np.zeros(len(train_dataSet))
    test_dataSet['f_prediction'] = np.zeros(len(train_dataSet))
    train_dataSet['prediction'] = np.zeros(len(train_dataSet))
    test_dataSet['prediction'] = np.zeros(len(train_dataSet))
    train_dataSet['miss'] = np.zeros(len(train_dataSet))
    test_dataSet['miss'] = np.zeros(len(train_dataSet))

    # Convert the "y" label to label and convert the yes or no to 1 or -1
    train_dataSet = train_dataSet.rename(columns={'y': 'label'})
    test_dataSet = test_dataSet.rename(columns={'y': 'label'})
    train_dataSet['label'] = train_dataSet['label'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
    test_dataSet['label'] = test_dataSet['label'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)

    # Convert the numerical features into binary by using the media of attribute values as the threshold
    for a in attributes:
        if attributes_types[a][0] == 'numeric':
            median = train_dataSet[train_dataSet[a] != 'unknown'][a].astype(float).median()
            train_dataSet.loc[train_dataSet[a].astype(float) > median, a] = median + 1
            train_dataSet.loc[train_dataSet[a].astype(float) <= median, a] = median - 1
            test_dataSet.loc[test_dataSet[a].astype(float) > median, a] = median + 1
            test_dataSet.loc[test_dataSet[a].astype(float) <= median, a] = median - 1
            attributes_types[a] = [median + 1, median - 1]

    # Set iteration times for begged tree algorithm
    T = 500
    print("We totally have ", T, " iterations")
    out = baggedTree.bagged_tree(train_dataSet, test_dataSet, attributes_types, T)
    print(out)

    # Display the second figure for the results
    x = range(1, T + 1)
    fig1 = plt.figure(1)
    ax1 = plt.axes()
    ax1.plot(x, out[0], c='b', label='Train Accuracy')
    ax1.plot(x, out[1], c='r', label='Test Accuracy')
    ax1.set_title("Bagged Tree")
    plt.legend()
    plt.show()


Question_2b()
