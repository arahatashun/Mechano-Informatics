#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for assignment
Hopfield Network
"""
import numpy as np
import random


class Hopfield_Network():
    """Hopfield Network"""

    def __init__(self):
        """Constructor

        :param num_of_pattern: number of patterns to memorized
        """
        self.theta = np.ones([25, 1]) * 0.0
        self.weight = np.zeros([5 * 5, 5 * 5])

    def train(self, *train):
        """ train weight Hebbian

        :param train: n times n train list
        """
        length = len(train)
        for i in range(length):
            train_flatten = np.ravel(train[i]).reshape([5 * 5, 1])
            self.weight = train_flatten @ train_flatten.T / length
        np.fill_diagonal(self.weight, 0)

    def potential_energy(self, input_flatten):
        """calculate lyapunov function

        :param input_flatten:input image (flatten)
        :return v:energy
        """
        v = -1 / 2 * input_flatten.T @ self.weight @ input_flatten
        v += self.theta.T @ input_flatten
        return v

    def recall(self, input, iter_num=100):
        """recall image from input

        :param input: 5 times 5 array
        :param iter_num: number of iterations
        :return:
        """
        input_flatten = np.ravel(input)
        energy_array = []
        for i in range(iter_num):
            index = np.random.randint(25)  # 0 to 24 random value
            energy_now = self.potential_energy(input_flatten)
            energy_array.append(energy_now)
            temp = np.copy(input_flatten)
            temp[index] = -1 if temp[index] == 1 else 1
            energy_candidate = self.potential_energy(temp)
            # print("now", energy_now)
            # print("candidate", energy_candidate)
            # print("---------------")
            if energy_candidate < energy_now:
                input_flatten = temp[:]

        recall_img = np.reshape(input_flatten, (5, 5))
        return recall_img, energy_array


image_1 = np.array(
    [[-1, -1, 1, -1, -1],
     [-1, -1, 1, -1, -1],
     [-1, -1, 1, -1, -1],
     [-1, -1, 1, -1, -1],
     [-1, -1, 1, -1, -1]])


def accuray(teacher, recalled):
    """caluculate recall accuracy

    :param teacher:teacher image
    :param recalled: recalled image
    :return:
    """
    match = (teacher == recalled)
    precision = np.sum(match) / 25 * 100
    print("precision", precision)


def test_1(noise):
    """experiment 1

    :param noise: number of bit to be changed 1~5
    :return:
    """
    hopfield = Hopfield_Network()
    hopfield.train(image_1)
    index = random.sample([i for i in range(25)], noise)
    test = np.ravel(np.copy(image_1))
    test[index] = -test[index]
    test = np.reshape(test, (5, 5))
    recall, _ = hopfield.recall(test)
    accuray(image_1, recall)


if __name__ == '__main__':
    test_1(int(input("noise")))
