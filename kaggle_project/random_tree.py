import math
import pandas as pd
import numpy as np


class Node:
    def __init__(self, _attribute):
        self.attribute = _attribute
        self.branches = []


class Branch:
    def __init__(self, att_type, node):
        self.att_type = att_type
        self.node = node


class RandomDecisionTree:

    def __init__(self, train_data, attribute_types, max_depth, sub_set_size):
        self.data = train_data
        self.sub_size = sub_set_size
        self.max_depth = max_depth
        self.gain = self.cal_entropy
        self.label = 'label'
        self.types = attribute_types
        self.root = self.id3(self.data, 0)

    # Calculate the Entropy
    def cal_entropy(self, set_A, attribute):
        Entropy = 0
        for l in pd.unique(set_A[self.label]):
            prob = set_A[set_A[self.label] == l][self.label].count() / set_A[self.label].count()
            Entropy += -prob * math.log2(prob)
        types = pd.unique(set_A[attribute])
        for t in types:
            set_ratio = set_A[set_A[attribute] == t][attribute].count() / set_A[attribute].count()
            sub_entropy = 0
            for l in pd.unique(set_A[self.label]):
                prob = set_A[set_A[attribute] == t].loc[set_A[self.label] == l, self.label].count() / \
                       set_A[set_A[attribute] == t][attribute].count()
                if prob != 0:
                    sub_entropy += -prob * math.log2(prob)
            Entropy -= set_ratio * sub_entropy
        return Entropy

    # Find the best attribute
    def choose_best_feature(self, df, attributes):

        best_value = -1
        best_attribute = ""

        for a in attributes:
            entropy = self.gain(df, a)
            if entropy > best_value:
                best_value = entropy
                best_attribute = a

        return best_attribute

    def id3(self, current_Set, depth):
        attributes = current_Set.columns[:-1]
        # Base case 1 - all examples have same label
        if current_Set[self.label].unique().size == 1:
            return Node(current_Set.iloc[0][self.label])
        # Base case 2 - return a leaf node with the most common label
        if attributes.size == 0 or depth >= self.max_depth:
            return Node(current_Set[self.label].mode()[0])

        sub_set = current_Set.sample(self.sub_size, replace=True, ignore_index=True)

        # Find the best attribute
        best_attribute = self.choose_best_feature(sub_set, attributes)
        root = Node(best_attribute)

        for tp in self.types[best_attribute]:
            # if S_v is empty, add leaf node with the most common value of label in S
            if current_Set[current_Set[best_attribute] == tp][self.label].count() == 0:
                root.branches.append(Branch(tp, Node(current_Set[self.label].mode()[0])))
            else:
                # By using recursive call to create branch
                root.branches.append(Branch(tp, self.id3(
                    current_Set[current_Set[best_attribute] == tp].drop(columns=best_attribute), depth + 1)))

        return root

    def testing(self, test_dataSet):

        # accuracy = 0
        prediction = np.zeros(len(test_dataSet))
        n_test = len(test_dataSet)

        for i in range(0, n_test):
            test_df = test_dataSet.iloc[i].copy()

            current = self.root
            while len(current.branches) != 0:
                att_type = test_df[current.attribute]
                for b in current.branches:
                    if b.att_type == att_type:
                        current = b.node
                        break

            prediction[i] = current.attribute

        test_dataSet['predict'] = prediction
        return test_dataSet
