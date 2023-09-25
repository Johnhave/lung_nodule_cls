import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy.stats import chi2
import pandas as pd


def Hosmer_Lemeshow_test(data, Q=10):
    '''
    data: dataframe format, with ground_truth label name is y,
                                 prediction value column name is y_hat
    '''
    data = data.sort_values('y_hat')
    data['Q_group'] = pd.qcut(data['y_hat'], Q)

    y_p = data['y'].groupby(data.Q_group).sum()
    y_total = data['y'].groupby(data.Q_group).count()
    y_n = y_total - y_p

    y_hat_p = data['y_hat'].groupby(data.Q_group).sum()
    y_hat_total = data['y_hat'].groupby(data.Q_group).count()
    y_hat_n = y_hat_total - y_hat_p

    hltest = (((y_p - y_hat_p) ** 2 / y_hat_p) + ((y_n - y_hat_n) ** 2 / y_hat_n)).sum()
    pval = 1 - chi2.cdf(hltest, Q - 2)

    # print('\n HL-chi2({}): {}, p-value: {}\n'.format(Q - 2, hltest, pval))
    print(pval)
    return pval


def get_p_value(pred, gt):
    pred = pred[np.newaxis, :]
    gt = gt[np.newaxis, :]
    data_all = np.concatenate((pred, gt), axis=0)
    data_all = data_all.transpose()
    data = pd.DataFrame(data_all, columns=['y_hat', 'y'])
    return Hosmer_Lemeshow_test(data, Q=10)


def plot_cali(label_idx, train_test, model_str):
    label_dict = {12: 'lvi',
                  13: 'vpi',
                  14: 'ln'}

    n_fold = 5

    plt.figure()

    xxx = np.linspace(0, 1, 2)
    plt.plot(xxx, xxx, '--', label='Perfectly calibrated')

    for fold_n in range(n_fold):
        pred_array = np.load('../model_evaluate_array/{}_{}_{}_pd_{}.npy'.format(label_dict[label_idx],
                                                                                 train_test,
                                                                                 model_str,
                                                                                 fold_n))
        gt_array = np.load('../model_evaluate_array/{}_{}_{}_gt_{}.npy'.format(label_dict[label_idx],
                                                                               train_test,
                                                                               model_str,
                                                                               fold_n))
        p_val = get_p_value(pred_array, gt_array)
        prob_gt, prob_pred = calibration_curve(gt_array, pred_array, n_bins=8)
        plt.plot(prob_pred, prob_gt, label='fold{}  p={:.4f}'.format(fold_n, p_val), marker='s')

    if train_test == 'train':
        title_str = 'Calibration plots (Reliability Curves) in training set'
    else:
        title_str = 'Calibration plots (Reliability Curves) in validation set'

    xlabel_dict = {'dl': 'DL', 'cl': 'Clinical', 'cldl': 'Combined'}
    plt.xlabel('{} Model Predicted Probability'.format(xlabel_dict[model_str]))
    plt.ylabel('Actual {} Rate'.format(label_dict[label_idx].upper()))
    plt.title(title_str)
    plt.legend()
    plt.savefig('../evaluate/{}_{}_cali_{}.tiff'.format(label_dict[label_idx], train_test, model_str))


if __name__ == '__main__':
    # train_test = ['train', 'test']
    train_test = ['val']
    model_str = ['cl', 'dl', 'cldl']
    label_idx = [12, 13, 14]

    for ll in label_idx:
        for tt in train_test:
            for mm in model_str:
                plot_cali(ll, tt, mm)
