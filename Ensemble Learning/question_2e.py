import pandas as pd
import numpy as np
import math
import decision_random_tree
import matplotlib.pyplot as plt
import randomForest


def Question_2e():
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

    # Set iteration times
    T = 4
    print("We totally have ", T, " iterations")

    # Through the bias and variance decomposition
    single_bias = 0
    single_variance = 0
    forest_bias = 0
    forest_variance = 0

    # repeat for 100 times
    for i in range(100):
        # sample 1000 examples uniformly without replacement
        s = train_dataSet.sample(1000, replace=False, ignore_index=True)

        # run random forest learning algorithm to learn 500 tree
        out = randomForest.rand_forest(s, test_dataSet, attributes_types, T, 6)

        # get dataset from begged tree algorithm
        test_data = out[2]
        first_test_data = out[3]
        forest_bias += sum((test_data['label']
                            - np.sign(test_data['f_prediction'])) ** 2) / len(test_data)
        single_bias += sum((first_test_data['label']
                            - np.sign(first_test_data['f_prediction'])) ** 2) / len(first_test_data)
        forest_mean = test_data['f_prediction'].mean()
        forest_variance += sum((forest_mean
                                - np.sign(test_data['f_prediction'])) ** 2) / (len(test_data) - 1)
        single_variance += sum((forest_mean
                                - np.sign(first_test_data['f_prediction'])) ** 2) / (len(first_test_data) - 1)

    # print results
    print("single tree bias", single_bias / len(test_dataSet))
    print("single tree variance", single_variance / len(test_dataSet))
    print("forest bias", forest_bias / len(test_dataSet))
    print("forest variance", forest_variance / len(test_dataSet))


Question_2e()
