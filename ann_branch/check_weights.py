import os

import torch
import shap
from model import ANN
import os
import numpy as np

if __name__ == '__main__':
    batch_size = 16
    label_idx = 14
    dataset_dir = '../dataset_extract'
    cl_feature_dir = '../../cl_feature_array'
    root_dir = '../../data_pool_sysucc_gy'
    label_dict = {12: 'lvi',
                  13: 'vpi',
                  14: 'ln'}
    n_fold = 5

    fn = open('feature_names.txt', 'r')
    feature_names = []
    for line in fn.readlines():
        feature_names.append(line.rstrip())

    dataX = []
    for file in os.listdir(cl_feature_dir):
        dataX.append(np.load(os.path.join(cl_feature_dir, file))[0])
    dataX = torch.from_numpy(np.array(dataX)).type(torch.FloatTensor)
    shap_values_all = []

    for fold_n in range(5):
        save_model_str = 'ann_bs{}_lb{}_wabs_fold{}'.format(batch_size, label_dict[label_idx], fold_n)

        # 创建网络模型
        check_point = torch.load('weights/{}.pth'.format(save_model_str))
        model = ANN(num_classes=1)
        model.load_state_dict(check_point)

        explainer = shap.DeepExplainer(model, dataX)
        shap_values = explainer.shap_values(dataX)
        shap_values_all.append(shap_values)
        print(save_model_str)

    shap_values_all = np.array(shap_values_all).mean(axis=0)
    shap.summary_plot(shap_values_all, dataX, feature_names=feature_names, max_display=33,
                      title='{} clinical model'.format(label_dict[label_idx]))
