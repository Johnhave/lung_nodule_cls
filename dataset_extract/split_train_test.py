# --coding:utf-8--
import random
import numpy as np


def generate_train_test_list(lab_0, lab_1, lab_idx):

    split_index_0 = round(len(lab_0) * split_factor) + 1
    split_index_1 = round(len(lab_1) * split_factor) + 1
    random.shuffle(lab_1)
    random.shuffle(lab_0)

    train_list_0 = lab_0[split_index_0:]
    train_list_1 = lab_1[split_index_1:]
    train_list = train_list_1 + train_list_0
    train_txt = open('train_data_{}.txt'.format(lab_idx), 'w')
    for line in train_list:
        train_txt.write(line)
    train_txt.close()

    test_list_0 = lab_0[:split_index_0]
    test_list_1 = lab_1[:split_index_1]
    test_list = test_list_1 + test_list_0
    test_txt = open('test_data_{}.txt'.format(lab_idx), 'w')
    for line in test_list:
        test_txt.write(line)
    test_txt.close()

split_factor = 0.2
np.random.seed(2023)

dataset_txt = open('dataset_all.txt', 'r')



lab12_0 = []
lab12_1 = []
lab13_0 = []
lab13_1 = []
lab14_0 = []
lab14_1 = []
for line in dataset_txt.readlines():
    file_dir, label12, label13, label14 = line.rstrip().split(',')
    if label12 == '1':
        lab12_1.append('{},{}\n'.format(file_dir, label12))
    else:
        lab12_0.append('{},{}\n'.format(file_dir, label12))

    if label13 == '1':
        lab13_1.append('{},{}\n'.format(file_dir, label13))
    else:
        lab13_0.append('{},{}\n'.format(file_dir, label13))

    if label14 == '1':
        lab14_1.append('{},{}\n'.format(file_dir, label14))
    else:
        lab14_0.append('{},{}\n'.format(file_dir, label14))
dataset_txt.close()

generate_train_test_list(lab12_0, lab12_1, 12)
generate_train_test_list(lab13_0, lab13_1, 13)
generate_train_test_list(lab14_0, lab14_1, 14)

