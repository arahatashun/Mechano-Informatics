#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for assignment
Hopfield Network
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import datasets

plt.rcParams['font.family'] = 'IPAPGothic'


def gini_index(df):
    """gini index"""
    num_data = df.shape[0]
    if num_data == 0:
        # print('numdate is 0')
        return 0
    gini = 0
    for i in np.unique(df['target']):
        df['target'] == i
        prob = len(df[df['target'] == i]) / num_data
        gini += prob * (1 - prob)
    return gini


def partial_gini(df, attribute, type):
    sample = df.sort_values(attribute).reset_index(drop=True)
    num_sample = len(sample)
    if type == 'greedy':
        split_index = int(num_sample / 2)
        # print('num_sample', num_sample)
        # print('split_index', split_index)
        threshold = 1 / 2 * (sample.loc[split_index - 1, attribute] + sample.loc[split_index, attribute])
        left = sample[sample[attribute] <= threshold]
        right = sample[sample[attribute] > threshold]
        left_gini = np.sum(sample[attribute] <= threshold) / num_sample * gini_index(left)
        right_gini = np.sum(sample[attribute] > threshold) / num_sample * gini_index(right)
        delat_gini = gini_index(sample) - (left_gini + right_gini)
        return delat_gini, threshold
    elif type == 'grid':
        delta_gini_list = np.zeros(num_sample - 1)
        threshold_list = np.zeros(num_sample - 1)
        for i in range(num_sample - 1):
            split_index = i + 1
            threshold = 1 / 2 * (sample.loc[split_index - 1, attribute] + sample.loc[split_index, attribute])
            left = sample[sample[attribute] <= threshold]
            right = sample[sample[attribute] > threshold]
            left_gini = np.sum(sample[attribute] <= threshold) / num_sample * gini_index(left)
            right_gini = np.sum(sample[attribute] > threshold) / num_sample * gini_index(right)
            delat_gini = gini_index(sample) - (left_gini + right_gini)
            delta_gini_list[i - 1] = delat_gini
            threshold_list[i - 1] = threshold
        max_index = np.argmax(delta_gini_list)
        return delta_gini_list[max_index], threshold_list[max_index]


def search_attribute(df, type):
    columns = df.columns
    max_delta_gini = 0.0
    threshold = None
    attribute = None
    for i in range(len(columns) - 1):
        delta_gini, tmp_threshold = partial_gini(df, columns[i], type)
        if delta_gini > max_delta_gini + sys.float_info.epsilon:
            min_delta_gini = delta_gini
            threshold = tmp_threshold
            attribute = columns[i]
    print("delta gini:", delta_gini)
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

    def split_node(self, type):
        attribute, threshold, gini = search_attribute(self.df, type)
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
        if self.left == None and self.right == None:
            return True
        else:
            # print("This is not a leaf node")
            return False

    def predict(self, test):
        if self.left is None and self.right is None:
            classification_result = self.df['target'].mode()[0]
            return classification_result
        else:
            # print("Go Deeper")
            if test[self.attribute] < self.threshold:
                return self.left.predict(test)
            else:
                return self.right.predict(test)

    def prune(self, epsilon):
        if self.isleaf():
            return
        self.left.prune(epsilon)
        self.right.prune(epsilon)
        if self.left.threshold == None and self.right.threshold == None:
            if self.gini < epsilon:
                print("Pruning")
                self.left = None
                self.right = None

    def build(self, type):
        """recursive call for building decision tree"""
        if len(self.df['target'].unique()) != 1:
            # print("build")
            left, right = self.split_node(type)
            self.left = left
            self.right = right
            self.left.build(type)
            self.right.build(type)
        else:
            return None


class DecisionTree():
    def __init__(self, df):
        self.root = Node(df)

    def train(self, type):
        self.root.build(type)

    def predict(self, example):
        return self.root.predict(example)

    def prune(self, epsilon):
        self.root.prune(epsilon)


def make_prefiction(df, epsilon, type, i):
    df_train = df.drop(df.index[i]).reset_index(drop=True)
    tree = DecisionTree(df_train)
    tree.train(type)
    tree.prune(epsilon)
    predicit_label = tree.predict(df.loc[i, :])
    answer_label = df.loc[i, 'target']
    # print("predict",predicit_label)
    # print("answer",answer_label)
    if predicit_label == answer_label:
        return 1
    else:
        return 0


def loo(df, epsilon, type='greedy'):
    """leave one out method"""
    r = Parallel(n_jobs=-1)([delayed(make_prefiction)(df, epsilon, type, i) for i in range(len(df))])
    accuracy = sum(r) / len(df)
    print("accuracy", accuracy)
    return accuracy


if __name__ == '__main__':
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target_names[iris.target]
    epsilon_list = np.linspace(0,1,10)
    greedy = np.zeros(10)
    grid = np.zeros(10)
    for i in range(len(epsilon_list)):
        greedy[i] = loo(df, epsilon_list[i])
        grid[i] = loo(df, epsilon_list[i], 'grid')
    plt.plot(epsilon_list, greedy, label= '２分割')
    plt.plot(epsilon_list, grid, label='ジニ不純度最大の分割')
    plt.xlabel("pruning")
    plt.ylabel("accuracy")
    plt.title("LOO")
    plt.legend()
    plt.savefig("loo.pdf")
    plt.show()

