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
import random

plt.rcParams['font.family'] = 'IPAPGothic'


class Hopfield_Network():
    """Hopfield Network"""

    def __init__(self):
        """Constructor

        :param num_of_pattern: number of patterns to memorized
        """
        self.theta = np.zeros([25, 1])
        self.weight = np.zeros([5 * 5, 5 * 5])

    def train(self, *train):
        """ train weight Hebbian

        :param train: n times n train list
        """
        length = len(train)
        for i in range(length):
            train_flatten = np.ravel(train[i]).reshape([5 * 5, 1])
            self.weight += train_flatten @ train_flatten.T / length  # ここは和では
        np.fill_diagonal(self.weight, 0)  # destructive function

    def potential_energy(self, input_flatten):
        """calculate lyapunov function

        :param input_flatten:input image (flatten)
        :return v:energy
        """
        v = 0
        """
        for i in range(len(input_flatten)):
            for j in range(len(input_flatten)):
                v += - 1/2 * self.weight[i][j] * input_flatten[i] * input_flatten[j]
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


def generate_images(num):
    """ number of image to generate

    :param num:
    :return:
    """
    images = []
    for i in range(num):
        def gen_img():
            image = np.random.rand(5, 5)
            image[image > 0.5] = 1
            image[~(image > 0.5)] = -1
            if np.sum([np.array_equal(image, images[j]) for j in range(i - 1)]) >= 1:
                image = gen_img()
            else:
                pass
            return image

        im = gen_img()
        images.append(im)

    return images


def check_ans(teacher, recalled):
    """caluculate recall similarity and correlct or not

    :param teacher:teacher image
    :param recalled: recalled image
    :return:
    """
    match = (teacher == recalled)
    similarity = np.sum(match) / 25 * 100
    complete_match = 0
    if similarity >= 100:
        complete_match = 1
    return similarity, complete_match


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
    EXP_NUM = 100
    sim_sum = 0
    correct_sum = 0
    for i in range(EXP_NUM):
        hopfield = Hopfield_Network()
        image = generate_images(1)
        hopfield.train(image)
        test = add_noise(image, noise)
        recall, energy = hopfield.recall(test)
        sim, correct = check_ans(image, recall)
        sim_sum += sim / EXP_NUM
        correct_sum += correct / EXP_NUM
    print("類似度", sim_sum, "正答率", correct_sum)
    return sim_sum, correct


def test_multi_img(noise, num_image):
    """experiment for multi image

    :param noise:
    :param num_image:number of image to remeber
    :return:
    """

    EXP_NUM = 100
    sim_sum = 0
    correct_sum = 0
    for i in range(EXP_NUM):
        hopfield = Hopfield_Network()
        images = generate_images(num_image)
        hopfield.train(*images)
        target_image = images[0]
        test = add_noise(target_image, noise)
        recall, energy = hopfield.recall(test)
        sim, correct = check_ans(target_image, recall)
        sim_sum += sim / EXP_NUM
        correct_sum += correct / EXP_NUM
    # print("類似度", sim_sum, "正答率", correct_sum)
    return sim_sum, correct_sum


def test_2():
    noise_list = np.arange(26)
    noise_percentage = noise_list * 4
    max_image_num = 6
    sim_arr = np.zeros((len(noise_list), max_image_num))
    cor_arr = np.zeros((len(noise_list), max_image_num))
    for num_image in range(max_image_num):
        num_image = num_image + 1
        for noise in noise_list:
            sim, cor = test_multi_img(noise, num_image)
            cor = cor * 100
            sim_arr[noise, num_image - 1] = sim
            cor_arr[noise, num_image - 1] = cor
    fig, axes = plt.subplots(3, 2)
    for i in range(max_image_num):
        axes.flat[i].plot(noise_percentage, sim_arr[:, i], label='類似度')
        axes.flat[i].plot(noise_percentage, cor_arr[:, i], label='正答率')
        axes.flat[i].legend()
        axes.flat[i].set_title("画像" + str(i + 1) + "枚")
    # plt.savefig("test2.pgf")
    np.save("sim", sim_arr)
    np.save("cor", cor_arr)
    plt.show()


def make_orthginal(num):
    images = []
    replace_index = random.sample(range(25), num)
    for i in range(num):
        im = np.ones((5,5))
        im = im.flatten()
        im[replace_index[i]] = -1
        im = im.reshape((5, 5))
        images.append(im)
    return images

def test_3(num_image=20):
    """

    :param num_image: #to remember
    :return:
    """
    noise_list = np.arange(26)
    noise_percentage = noise_list * 4
    sim_arr = np.zeros((len(noise_list)))
    cor_arr = np.zeros((len(noise_list)))
    for noise in noise_list:
        EXP_NUM = 100
        sim_sum = 0
        correct_sum = 0
        for i in range(EXP_NUM):
            hopfield = Hopfield_Network()
            images = make_orthginal(num_image)
            hopfield.train(*images)
            target_image = images[0]
            test = add_noise(target_image, noise)
            recall, energy = hopfield.recall(test)
            sim, correct = check_ans(target_image, recall)
            sim_sum += sim / EXP_NUM
            correct_sum += correct / EXP_NUM
        sim_arr[noise] = sim_sum
        cor_arr[noise] = correct_sum * 100
    cor_rand =  np.zeros((len(noise_list)))
    sim_rand = np.zeros((len(noise_list)))
    for noise in noise_list:
        sim, cor = test_multi_img(noise, num_image)
        cor = cor * 100
        sim_arr[noise] = sim
        cor_arr[noise] = cor
    fig, ax = plt.subplots(1, 1)
    ax.plot(noise_percentage, sim_arr, label='類似度(直交性有)')
    ax.plot(noise_percentage, cor_arr, label='正答率(直交性有)')
    ax.plot(noise_percentage, sim_rand, label='類似度')
    ax.plot(noise_percentage, cor_rand, label='正答率')
    ax.legend()
    ax.set_xlabel("Noise")
    # np.save("test_3_sim", sim_arr)
    # np.save("test_3_cor", cor_arr)
    plt.savefig("test3.pgf")
    plt.show()


if __name__ == '__main__':
    # test_1(int(input("noise")))
    # test2()
    test_3()
