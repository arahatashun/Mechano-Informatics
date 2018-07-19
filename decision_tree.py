#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for assignment
Hopfield Network
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets

plt.rcParams['font.family'] = 'IPAPGothic'


def gini_index(df):
    """gini index"""
    num_data = df.shape[0]
    if num_data == 0:
        print('numdate is 0')
        return 0
    gini = 0
    for i in np.unique(df['target']):
        df['target'] == i
        prob = len(df[df['target'] == i]) / num_data
        gini += prob * (1 - prob)
    return gini


def partial_gini(df, attribute):
    sample = df.sort_values(attribute).reset_index(drop=True)
    num_sample = len(sample)
    split_index = int(num_sample / 2)
    print('num_sample', num_sample)
    print('split_index', split_index)
    threshold = 1 / 2 * (sample.loc[split_index - 1, attribute] + sample.loc[split_index, attribute])
    left = sample[sample[attribute] <= threshold]
    right = sample[sample[attribute] > threshold]
    left_gini = np.sum(sample[attribute] <= threshold) / num_sample * gini_index(left)
    right_gini = np.sum(sample[attribute] > threshold) / num_sample * gini_index(right)
    delat_gini = gini_index(sample) - (left_gini + right_gini)
    return delat_gini, threshold


def search_attribute(df):
    columns = df.columns
    max_delta_gini = 0
    threshold = None
    attribute = None
    for i in range(len(columns) - 1):
        delta_gini, tmp_threshold = partial_gini(df, columns[i])
        print(delta_gini)
        if delta_gini > max_delta_gini:
            min_delta_gini = delta_gini
            threshold = tmp_threshold
            attribute = columns[i]
    return attribute, threshold, delta_gini


class Node(object):
    def __init__(self, df):
        self.left = None
        self.right = None
        self.attribute = None
        self.threshold = None
        self.df = df
        self.members = df.shape[0]
        assert not self.df.empty, "DateFrame Empty"

    def split_node(self):
        attribute, threshold, gini = search_attribute(self.df)
        # print("split attribute:", attribute)
        # print("split threshold:", threshold)
        self.left = Node(self.df[self.df[attribute] <= threshold])
        self.right = Node(self.df[self.df[attribute] > threshold])
        self.attribute = attribute
        self.threshold = threshold
        self.gini = gini
        # print("left node:", self.left.members)
        # print("right node:", self.right.members)
        return self.left, self.right

    def isleaf(self):
        if len(self.df['target'].unique()) == 1:
            print("This is a leaf node")
            return True
        elif self.left == None and self.right == None:
            return True
        else:
            print("This is not a leaf node")
            return False

    def predict(self, test):
        if self.left is None and self.right is None:
            classification_result = self.df['target'].mode()[0]
            return classification_result
        else:
            print("Go Deeper")
            if test[self.attribute] < self.threshold:
                return self.left.predict(test)
            else:
                return self.right.predict(test)

    def prune(self, epsilon):
        if self.isleaf(): return
        self.left.prune(epsilon)
        self.right.prune(epsilon)
        if self.left == None and self.right == None:
            if self.gini < epsilon:
                print('Pruning')
                self.left = None
                self.right = None


def build_decision_tree(root, Node_list):
    """recursive call for building decision tree"""
    if root.isleaf() is False:
        print('function called')
        left, right = root.split_node()
        Node_list.append(left)
        Node_list.append(right)
        build_decision_tree(left, Node_list)
        build_decision_tree(right, Node_list)
    else:
        return None


class DecisionTree():
    def __init__(self, df):
        self.nodes = []
        self.root = Node(df)

    def train(self):
        build_decision_tree(self.root, self.nodes)

    def predict(self, example):
        return self.root.predict(example)

    def prune(self, epsilon):
        self.root.prune(epsilon)


if __name__ == '__main__':
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target_names[iris.target]
    np.random.seed(0)
    msk = np.random.rand(len(df)) < 0.8
    df_train = df[msk].reset_index(drop=True)
    df_test = df[~msk].reset_index(drop=True)
    tree = DecisionTree(df_train)
    tree.train()
    tree.prune(20)
    right_prediction = 0
    for i in range(len(df_test)):
        predicit_label = tree.predict(df_test.loc[i, :])
        answer_label = df_test.loc[i, 'target']
        if predicit_label == answer_label:
            right_prediction += 1

    print("accuracy", right_prediction / len(df_test))
