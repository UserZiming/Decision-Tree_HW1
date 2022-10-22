import pandas as pd
import numpy as np
import math
from pandas.core.frame import DataFrame

class Node:
    def __init__(self, _attribute):
        self.attribute = _attribute
        self.branches = []

class Branch:
    def __init__(self, att_type, node):
        self.node = node
        self.att_type = att_type

class DecisionTree:
    def __init__(self, train_dataSet, attribute_types, max_depth, sub_size):
        # Add new variable for modifying decision tree
        self.sub_size = sub_size
        self.gain = self.cal_entropy
        self.data = train_dataSet
        self.label = self.data.columns[-1]
        self.types = attribute_types
        self.max_depth = max_depth

        self.root = self.id3(self.data, 0)

    # Calculate the Entropy
    def cal_entropy(self, dataSet, attribute):
        Entropy = 0
        column_labels = pd.unique(dataSet[self.label])

        for lb in column_labels:

            n_labels = dataSet[self.label].count()
            match_label = dataSet[dataSet[self.label] == lb][self.label].count()

            prob = match_label/n_labels
            Entropy += -prob*math.log2(prob)

        types = pd.unique(dataSet[attribute])

        for t in types:

            n_attributes = dataSet[attribute].count()
            match_attribute = dataSet[dataSet[attribute] == t][attribute].count()
            set_ratio = match_attribute/n_attributes

            sub_Entropy = 0
            for lb in column_labels:

                match_attribute = dataSet[dataSet[attribute] == t][attribute].count()
                match_att_label = dataSet[dataSet[attribute] == t].loc[dataSet[self.label] == lb, self.label].count()

                prob = match_att_label / match_attribute

                if prob != 0:
                    sub_Entropy += -prob*math.log2(prob)
            Entropy -= set_ratio*sub_Entropy

        return Entropy

    # Find the best attribute
    def choose_best_feature(self, df, attributes):

        subSet = df.sample(self.sub_size, replace=True, ignore_index=True)

        best_value = -1
        best_attribute = ""

        for a in attributes:
            entropy = self.gain(subSet, a)
            if entropy > best_value:
                best_value = entropy
                best_attribute = a

        return best_attribute

    def id3(self, dataset, depth):
        attributes = dataset.columns[:-1]
        # Base case 1 - all examples have same label
        if dataset[self.label].unique().size == 1:
            return Node(dataset.iloc[0][self.label])

        # Base case 2 - return a leaf node with the most common label
        if attributes.size == 0 or depth >= self.max_depth:
            return Node(dataset[self.label].mode()[0])

        # Find the best attribute
        best_attribute = self.choose_best_feature(dataset, attributes)
        root = Node(best_attribute)

        for tp in self.types[best_attribute]:
            # if S_v is empty, add leaf node with the most common value of label in S
            if dataset[dataset[best_attribute] == tp][self.label].count() == 0:
                root.branches.append(Branch(tp, Node(dataset[self.label].mode()[0])))
            else:
                # By using recursive call to create branch
                root.branches.append(Branch(tp, self.id3(dataset[dataset[best_attribute] == tp].drop(columns=best_attribute), depth+1)))

        return root

    def testing(self, test_dataSet):

        # accuracy = 0
        pred = np.zeros(len(test_dataSet))
        miss = np.zeros(len(test_dataSet))
        col_labels = test_dataSet[self.label].count()

        for i in range(0, col_labels):
            test_df = test_dataSet.iloc[i].copy()

            current = self.root
            while len(current.branches) != 0:
                att_type = test_df[current.attribute]
                for b in current.branches:
                    if b.att_type == att_type:
                        current = b.node
                        break

            pred[i] = current.attribute
            if test_df[self.label] == current.attribute:
                miss[i] = 0
            else:
                miss[i] = 1
        test_dataSet['prediction'] = pred
        test_dataSet['miss'] = miss

        # accuracy = accuracy / len(test_dataSet.columns)

        return test_dataSet
