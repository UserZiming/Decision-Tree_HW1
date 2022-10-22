import pandas as pd
import numpy as np
from math import log
from pandas.core.frame import DataFrame


class Node:
    def __init__(self, _attribute):
        self.attribute = _attribute
        self.branches = []


class Branch:
    def __init__(self, attribute_type, node):
        self.node = node
        self.attribute_type = attribute_type


class DecisionTree:
    Entropy_ID = 0
    Majority_Error_ID = 1
    Gini_Index_ID = 2

    def __init__(self, train_dataSet, attribute_types, info_gain_type, max_depth):
        self.data = train_dataSet
        self.label = self.data.columns[-1]
        self.types = attribute_types
        self.max_depth = max_depth
        # self.cur_depth = cur_depth

        if info_gain_type == self.Entropy_ID:
            self.gain = self.cal_entropy
        if info_gain_type == self.Majority_Error_ID:
            self.gain = self.majority_error
        if info_gain_type == self.Gini_Index_ID:
            self.gain = self.gini_index

        self.root = self.id3(self.data, 0)

    # Calculate the Entropy
    def cal_entropy(self, dataSet):
        n_rows = len(dataSet)
        col_label = dataSet.columns[-1]
        n_labels = {}

        # for featVec in dataSet:
        for val in dataSet[col_label]:
            # Get the last column of data
            current_lable = val

            if current_lable not in n_labels.keys():
                n_labels[current_lable] = 0

            n_labels[current_lable] += 1

        Entropy = 0
        for i in n_labels:
            if n_rows == 0:
                probability = 0
            else:
                probability = float(n_labels[i]) / n_rows
            Entropy -= probability * log(probability, 2)

        return Entropy

    def majority_error(self, dataSet):
        n_rows = len(dataSet)
        col_label = dataSet.columns[-1]
        n_labels = {}

        # for featVec in dataSet:
        for val in dataSet[col_label]:
            # Get the last column of data
            current_lable = val

            if current_lable not in n_labels.keys():
                n_labels[current_lable] = 0

            n_labels[current_lable] += 1

        ME = 0
        max_val = max(n_labels.items(), key=lambda x: x[1])[0]
        max_number = max(n_labels.items(), key=lambda x: x[1])[1]
        total = 0
        for val in n_labels.values():
            total += val
        ME = max_number / total

        return ME

    # Calculate gini index
    def gini_index(self, dataSet):
        n_rows = len(dataSet)
        col_label = dataSet.columns[-1]
        n_labels = {}

        # for featVec in dataSet:
        for val in dataSet[col_label]:
            # Get the last column of data
            current_lable = val

            if current_lable not in n_labels.keys():
                n_labels[current_lable] = 0

            n_labels[current_lable] += 1

        GI = 0
        for i in n_labels:
            if n_rows == 0:
                probability = 0
            else:
                probability = float(n_labels[i]) / n_rows
            GI = 1 - probability * probability

        return GI

    def split_dataset(self, dataset, attributes, value):
        sub_dataset = dataset[dataset[attributes] == value]
        sub_dataset.drop([attributes], axis=1)

        # Returns a subset without the partition feature
        return sub_dataset

    def choose_best_feature(self, df):
        features = df.columns
        # base_entropy = df.cal_entropy(df)
        base_entropy = self.cal_entropy(df)
        best_feature = ''
        best_info_gain = 0

        for index, col in enumerate(features):
            #if col == 'label' | col == self.label:
            # if col == self.label:
            #     break
            # else:
            # Get all values of a feature (a column)
            feature_list = df[col].to_list()
            # No duplicate attribute eigenvalues
            features_unique = set(feature_list)
            new_entropy = 0
            for value in features_unique:
                # sub_df = df.split_dataset(df, features[index], value)
                sub_df = self.split_dataset(df, features[index], value)
                prob = (len(sub_df)) / len(df)
                # new_entropy += prob * df.cal_entropy(sub_df)
                new_entropy += prob * self.cal_entropy(sub_df)
            info_gain = base_entropy - new_entropy

            if best_info_gain < info_gain:
                best_feature = col

        return best_feature

    def id3(self, df, depth):
        # Base case 1 - all examples have same label
        if df[self.label].unique().size == 1:
            leaf_node = Node(df.iloc[0][self.label])
            return leaf_node

        # Base case 2 - return a leaf node with the most common label
        if depth >= self.max_depth:
            leaf_node = Node(df[self.label].mode()[0])
            return leaf_node

        attributes = df.columns[:-1]

        if attributes.size == 0:
            return Node(df[self.label].mode()[0])

        temp_df = df
        temp_df = temp_df.drop([self.label], axis=1)
        best_feature = self.choose_best_feature(temp_df)

        # Set that as the feature for the root
        root_node = Node(best_feature)

        print(type(self.types))
        print(self.types[best_feature])
        print(best_feature)
        types = self.types[best_feature]


        for t in types:
            if df[df[best_feature] == t][self.label].count() == 0:
                # if S_v is empty, add leaf node with the most common value of label in S
                B = Branch(t, Node(df[self.label].mode()[0]))
                root_node.branches.append(B)
            else:
                # By using recursive call to create branch
                root_node.branches.append(
                    Branch(t, self.id3(df[df[best_feature] == t].drop(columns=best_feature, axis=1), depth + 1)))

        return root_node

    def testing(self, test_dataSet):
        accuracy = 0
        for i in range(0, len(test_dataSet.columns)):
            col = test_dataSet.iloc[i]
            cur_node = self.root
            while len(cur_node.branches) != 0:
                cur_attribute = col[cur_node.attribute]
                for branch in cur_node.branches:
                    if branch.attribute_type == cur_attribute:
                        cur_node = branch.node
                        break
            if col[self.label] == cur_node.attribute:
                accuracy += 1

        accuracy = accuracy / len(test_dataSet.columns)

        return accuracy


def make_report(train_dataSet, test_dataSet, attributes_types, depth):

    report = np.zeros((6, depth))

    # Use the entropy to calculate information gain and the depth is 6
    for i in range(1, depth+1):
        print(i)
        tree = DecisionTree(train_dataSet, attributes_types, DecisionTree.Entropy_ID, i)
        # Testing after get the decision tree
        report[i - 1, 0] = tree.testing(train_dataSet)
        report[i - 1, 1] = tree.testing(test_dataSet)

    # Use the majority error to calculate information gain and the depth is 6
    for i in range(1, depth+1):
        tree = DecisionTree(train_dataSet, attributes_types, DecisionTree.Majority_Error_ID, i)
        # Testing after get the decision tree
        report[i - 1, 2] = tree.testing(train_dataSet)
        report[i - 1, 3] = tree.testing(test_dataSet)

    # Use the gini index to calculate information gain and the depth is 6
    for i in range(1, depth+1):
        tree = DecisionTree(train_dataSet, attributes_types, DecisionTree.Gini_Index_ID, i)
        # Testing after get the decision tree
        report[i - 1, 4] = tree.testing(train_dataSet)
        report[i - 1, 5] = tree.testing(test_dataSet)

    return report


def Question_2():
    train_path = './car/train.csv'
    test_path = './car/test.csv'

    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

    train_dataSet = pd.read_csv(train_path, header=None, names=attributes)
    test_dataSet = pd.read_csv(test_path, header=None, names=attributes)

    attributes_types = {}
    for index, col in enumerate(attributes):
        values = set(train_dataSet[col])
        attributes_types[attributes[index]] = values
    print(attributes_types)

    # report = np.zeros((6, 6))
    #
    # # Use the entropy to calculate information gain and the depth is 6
    # for i in range(1, 7):
    #     tree = DecisionTree(train_dataSet, attributes_types, DecisionTree.Entropy_ID, i)
    #     # Testing after get the decision tree
    #     report[i - 1, 0] = tree.testing(train_dataSet)
    #     report[i - 1, 1] = tree.testing(test_dataSet)
    #
    # # Use the majority error to calculate information gain and the depth is 6
    # for i in range(1, 7):
    #     tree = DecisionTree(train_dataSet, attributes_types, DecisionTree.Majority_Error_ID, i)
    #     # Testing after get the decision tree
    #     report[i - 1, 2] = tree.testing(train_dataSet)
    #     report[i - 1, 3] = tree.testing(test_dataSet)
    #
    # # Use the gini index to calculate information gain and the depth is 6
    # for i in range(1, 7):
    #     tree = DecisionTree(train_dataSet, attributes_types, DecisionTree.Gini_Index_ID, i)
    #     # Testing after get the decision tree
    #     report[i - 1, 4] = tree.testing(train_dataSet)
    #     report[i - 1, 5] = tree.testing(test_dataSet)

    report = make_report(train_dataSet, test_dataSet, attributes_types, 6)

    print('\nThe accuracy table for car dataset with depth [1,6].')
    report_df = pd.DataFrame(report, columns=["entropy_train", "entropy_test", "me_train", "me_test", "gini_train",
                                              "gini_test"])
    report_df.insert(loc=0, column="depth", value=np.arange(1, 7))

    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(report_df, '\n')


def Question_3():
    train_path = './bank/train.csv'
    test_path = './bank/test.csv'

    attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                  'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

    train_dataSet = pd.read_csv(train_path, header=None, names=attributes)
    test_dataSet = pd.read_csv(test_path, header=None, names=attributes)

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
        'month': ["jan", "feb", "mar", "apr", "may", "jun","jul", "aug", "sep", "oct", "nov", "dec"],
        'duration': ['numeric'],
        'campaign': ['numeric'],
        'pdays': ['numeric'],
        'previous': ['numeric'],
        'poutcome': ["unknown", "other", "failure", "success"],
        'y': ['yes', 'no']
    }

    # Question (3a)
    report = make_report(train_dataSet, test_dataSet, attributes_types, 16)

    print('\nThe accuracy table for car dataset with depth [1,16].')
    report_df = pd.DataFrame(report, columns=["entropy_train", "entropy_test", "me_train", "me_test", "gini_train",
                                              "gini_test"])
    report_df.insert(loc=0, column="depth", value=np.arange(1, 17))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(report_df, '\n')


    # Question (3b) Consider “unknown” as a particular attribute value
    for a in attributes:
        if attributes_types[a][0] == 'numeric':
            median = train_dataSet[train_dataSet[a] != 'unknown'][a].astype(float).median()

            #attributes_types[a] = [median+1, median-1]
            attributes_types[a] = [median]

    report = make_report(train_dataSet, test_dataSet, attributes_types, 16)

    print('\nThe accuracy table for car dataset with depth [1,16].')
    report_df = pd.DataFrame(report, columns=["entropy_train", "entropy_test", "me_train", "me_test", "gini_train",
                                              "gini_test"])
    report_df.insert(loc=0, column="depth", value=np.arange(1, 17))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(report_df, '\n')



Question_2()
#Question_3()
