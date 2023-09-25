# import os
# import numpy as np
# import rpy2.robjects as robj
# r = robj.r
# from rpy2.robjects.packages import importr
#
#
# def roc_test_r(targets_1, scores_1, targets_2, scores_2, method='delong'):
#     # method: “delong”, “bootstrap” or “venkatraman”
#     # r('install.packages("pROC")')
#     importr('pROC')
#     robj.globalenv['targets_1'] = targets_1 = robj.FloatVector(targets_1)
#     robj.globalenv['scores_1'] = scores_1 = robj.FloatVector(scores_1)
#     robj.globalenv['targets_2'] = targets_2 = robj.FloatVector(targets_2)
#     robj.globalenv['scores_2'] = scores_2 = robj.FloatVector(scores_2)
#
#     r('roc_1 <- roc(targets_1, scores_1)')
#     r('roc_2 <- roc(targets_2, scores_2)')
#     r('result = roc.test(roc_1, roc_2, method="%s")' % method)
#     p_value = r('p_value = result$p.value')
#     return np.array(p_value)[0]
import numpy as np
import scipy.stats as st


class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        # self._show_result()
        self.zp = self._compute_z_p()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2

        return z, p

    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold:
            print("There is a significant difference")
        else:
            print("There is NO significant difference")


def get_z_p(preds_A, preds_B, actual):
    return DelongTest(preds_A, preds_B, actual).zp


if __name__ == '__main__':

    label_idx = 14  # [12, 13, 14]
    label_dict = {12: 'lvi',
                  13: 'vpi',
                  14: 'ln'}
    for label_idx in [12, 13, 14]:
        # for train_test in ['train', 'test']:
        for train_test in ['val']:
            z_cl_dl = []
            p_cl_dl = []
            z_cl_cldl = []
            p_cl_cldl = []
            z_dl_cldl = []
            p_dl_cldl = []
            for fold_n in range(5):
                y_pred_cl = np.load('../model_evaluate_array/{}_{}_{}_pd_{}.npy'.format(label_dict[label_idx],
                                                                                        train_test,
                                                                                        'cl',
                                                                                        fold_n))
                y_label = np.load('../model_evaluate_array/{}_{}_{}_gt_{}.npy'.format(label_dict[label_idx],
                                                                                      train_test,
                                                                                      'cl',
                                                                                      fold_n))
                y_pred_dl = np.load('../model_evaluate_array/{}_{}_{}_pd_{}.npy'.format(label_dict[label_idx],
                                                                                        train_test,
                                                                                        'dl',
                                                                                        fold_n))

                y_pred_cldl = np.load('../model_evaluate_array/{}_{}_{}_pd_{}.npy'.format(label_dict[label_idx],
                                                                                          train_test,
                                                                                          'cldl',
                                                                                          fold_n))

                z, p = get_z_p(y_pred_cl, y_pred_dl, y_label)
                z_cl_dl.append(z)
                p_cl_dl.append(p)

                z1, p1 = get_z_p(y_pred_cl, y_pred_cldl, y_label)
                z_cl_cldl.append(z1)
                p_cl_cldl.append(p1)

                z2, p2 = get_z_p(y_pred_dl, y_pred_cldl, y_label)
                z_dl_cldl.append(z2)
                p_dl_cldl.append(p2)

            print('{} in {}'.format(label_dict[label_idx], train_test))
            print('cl vs dl (z-score, p): {}, {}'.format(np.mean(z_cl_dl), np.mean(p_cl_dl)))
            print('cl vs cldl (z-score, p): {}, {}'.format(np.mean(z_cl_cldl), np.mean(p_cl_cldl)))
            print('dl vs cldl (z-score, p): {}, {}'.format(np.mean(z_dl_cldl), np.mean(p_dl_cldl)))
