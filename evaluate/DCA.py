# --coding:utf-8--
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


if __name__ == '__main__':
    thresh_group = np.arange(0, 0.4, 0.001)


    # train_test = 'train'  # ['train', 'test']
    # model_str = #['cl', 'dl', 'cldl']
    label_idx = 14  # [12, 13, 14]
    label_dict = {12: 'lvi',
                  13: 'vpi',
                  14: 'ln'}
    for label_idx in [12, 13, 14]:
        # for train_test in ['train', 'test']:
        for train_test in ['val']:
            for fold_n in range(5):
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                y_pred_score = np.load('../model_evaluate_array/{}_{}_{}_pd_{}.npy'.format(label_dict[label_idx],
                                                                                           train_test,
                                                                                           'cl',
                                                                                           fold_n))
                y_label = np.load('../model_evaluate_array/{}_{}_{}_gt_{}.npy'.format(label_dict[label_idx],
                                                                                      train_test,
                                                                                      'cl',
                                                                                      fold_n))
                net_benefit_model1 = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
                net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)

                y_pred_score = np.load('../model_evaluate_array/{}_{}_{}_pd_{}.npy'.format(label_dict[label_idx],
                                                                                           train_test,
                                                                                           'dl',
                                                                                           fold_n))
                y_label = np.load('../model_evaluate_array/{}_{}_{}_gt_{}.npy'.format(label_dict[label_idx],
                                                                                      train_test,
                                                                                      'dl',
                                                                                      fold_n))
                net_benefit_model2 = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

                y_pred_score = np.load('../model_evaluate_array/{}_{}_{}_pd_{}.npy'.format(label_dict[label_idx],
                                                                                           train_test,
                                                                                           'cldl',
                                                                                           fold_n))
                y_label = np.load('../model_evaluate_array/{}_{}_{}_gt_{}.npy'.format(label_dict[label_idx],
                                                                                      train_test,
                                                                                      'cldl',
                                                                                      fold_n))
                net_benefit_model3 = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

                # Plot
                y2 = np.maximum(net_benefit_all, 0)
                y1 = np.maximum(net_benefit_model1, y2)
                ax.plot(thresh_group, net_benefit_model1, color='r', label='Clinical Model')
                ax.fill_between(thresh_group, y1, y2, color='r', alpha=0.2)
                y2 = np.maximum(net_benefit_all, 0)
                y1 = np.maximum(net_benefit_model2, y2)
                ax.plot(thresh_group, net_benefit_model2, color='g', label='DL Model')
                ax.fill_between(thresh_group, y1, y2, color='g', alpha=0.2)
                y2 = np.maximum(net_benefit_all, 0)
                y1 = np.maximum(net_benefit_model3, y2)
                ax.plot(thresh_group, net_benefit_model3, color='b', label='Combined Model')
                ax.fill_between(thresh_group, y1, y2, color='b', alpha=0.2)
                ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
                ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

                # Fill，显示出模型较于treat all和treat none好的部分

                # Figure Configuration， 美化一下细节
                ax.set_xlim(0, 0.3)
                ax.set_ylim(-0.15, 0.2)
                # ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
                ax.set_xlabel(
                    xlabel='Threshold Probability',
                    fontdict={'family': 'Times New Roman', 'fontsize': 15}
                )
                ax.set_ylabel(
                    ylabel='Net Benefit',
                    fontdict={'family': 'Times New Roman', 'fontsize': 15}
                )
                ax.grid('major')
                ax.spines['right'].set_color((0.8, 0.8, 0.8))
                ax.spines['top'].set_color((0.8, 0.8, 0.8))
                ax.legend(loc='upper right')

                if train_test == 'train':
                    title_str = 'Fold {} in the training set'.format(fold_n)
                else:
                    title_str = 'Fold {} in the validation set'.format(fold_n)
                ax.set_title(title_str)

                fig.savefig('{}_{}_dca_fold{}.tiff'.format(label_dict[label_idx], train_test, fold_n))
                # plt.show()
