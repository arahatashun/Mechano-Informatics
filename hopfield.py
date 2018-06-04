#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for assignment
Hopfield Network
"""
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'IPAPGothic'



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
            self.weight += train_flatten @ train_flatten.T / length  #ここは和では
        np.fill_diagonal(self.weight, 0) # destructive function

    def potential_energy(self, input_flatten):
        """calculate lyapunov function

        :param input_flatten:input image (flatten)
        :return v:energy
        """
        v = -1 / 2 * input_flatten.T @ self.weight @ input_flatten
        v += self.theta.T @ input_flatten
        return v

    def recall(self, input, iter_num=1000):
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

image_2 = np.array(
    [[1, -1, -1, -1, -1],
     [-1, 1, -1, -1, -1],
     [-1, -1, 1, -1, -1],
     [-1, -1, -1, 1, -1],
     [-1, -1, -1, -1, 1]])

image_3 = np.array(
    [[-1, -1, -1, -1, 1],
     [-1, -1, -1, 1, -1],
     [-1, -1, 1, -1, -1],
     [-1, 1, -1, -1, -1],
     [1, -1, -1, -1, -1]])

image_4 = np.array(
    [[1, -1, -1, -1, -1],
     [1, -1, -1, -1, -1],
     [1, -1, -1, -1, -1],
     [1, -1, -1, -1, -1],
     [1, -1, -1, -1, -1]])

image_5 = np.array(
    [[-1, -1, -1, -1, 1],
     [-1, -1, -1, -1, 1],
     [-1, -1, -1, -1, 1],
     [-1, -1, -1, -1, 1],
     [-1, -1, -1, -1, 1]])

image_6 = np.array(
    [[-1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1],
     [1, 1, -1, 1, 1]])


def accuray(teacher, recalled):
    """caluculate recall accuracy

    :param teacher:teacher image
    :param recalled: recalled image
    :return:
    """
    match = (teacher == recalled)
    precision = np.sum(match) / 25 * 100
    print("precision", precision)


def add_noise(img, noise):
    """add noise to img

    :param img:
    :param noise: number of bit to be changed
    """
    index = random.sample([i for i in range(25)], noise)
    noisy_img = np.ravel(np.copy(img))
    noisy_img[index] = -1 * noisy_img[index]
    noisy_img = np.reshape(noisy_img, (5, 5))
    return noisy_img


def test_1(noise):
    """experiment 1

    :param noise: number of bit to be changed 1~5
    :return:
    """
    hopfield = Hopfield_Network()
    hopfield.train(image_1)
    test = add_noise(image_1, noise)
    print(test)
    recall, energy = hopfield.recall(test)
    accuray(image_1, recall)
    print(recall)
    return energy

def test_2(noise):
    """experiment 2

    :param noise:
    :return:
    """
    hopfield = Hopfield_Network()
    hopfield.train(image_1,image_2,image_3,image_4,image_5,image_6)
    test = add_noise(image_6, noise)
    recall, energy = hopfield.recall(test)
    accuray(image_3, recall)
    return energy

if __name__ == '__main__':
    ene1 = test_2(int(input("noise")))
    ene2 = test_2(int(input("noise")))
    plt.figure(figsize=(6, 5), facecolor="w")
    plt.plot(ene1)
    plt.plot(ene2)
    plt.show()
