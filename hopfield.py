#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for assignment
Hopfield Network
"""
import numpy as np


class Hopfield_Network():
    """Hopfield Network"""

    def __init__(self, num_of_pattern=1):
        """Constructor

        :param num_of_pattern: number of patterns to memorized
        """
        self.theta = np.zeros([25, 1])
        self.q = num_of_pattern
        self.weight = np.zeros([5 * 5, 5 * 5])

    def train(self, train_array):
        """ train weight Hebbian

        :param train_array: n times n
        """
        train_flatten = np.ravel(train_array)
        self.weight += train_flatten @ train_flatten.T / self.q
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
            temp = input_flatten[:]
            temp[index] = 1 if temp[index] == 0 else 1
            energy_candidate = self.potential_energy(temp)
            if energy_candidate < energy_now:
                input_flatten = temp

        recall_img = np.reshape(input_flatten, (5, 5))
        return recall_img, energy_array


image_1 = np.array(
    [[0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 1, 0, 0]])


def test_1():
    hopfield = Hopfield_Network()
    hopfield.train(image_1)
    test = np.array([[0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0]])
    hoge, _ = hopfield.recall(test)
    print(hoge)

if __name__ == '__main__':
    test_1()