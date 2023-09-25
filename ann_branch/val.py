# --coding:utf-8--
import torch
from torch.utils.data import DataLoader

from dataset import Lungdataset
from model import ANN
import numpy as np


def get_auc_and_ci(label0, label1):

    auclist = []

    nauc = 500
    for i in range(nauc):
        sublab0 = []
        sublab1 = []
        for lab0 in label0:
            if round(np.random.uniform(0, 1), 1) <= 0.7:
                sublab0.append(lab0)
        for lab1 in label1:
            if round(np.random.uniform(0, 1), 1) <= 0.7:
                sublab1.append(lab1)
        sublab0 = np.array(sublab0)
        sublab1 = np.array(sublab1)
        sumAll = 0
        for lab1 in sublab1:
            sumAll = sumAll + np.sum(sublab0 < lab1) + np.sum(sublab0 == lab1) / 2
        auc = sumAll / (sublab0.shape[0] * sublab1.shape[0])
        auclist.append(auc)

    auclist = np.array(auclist)
    mean = np.mean(auclist)
    std = np.std(auclist)

    print('auc: ', mean, '  CI: ', mean-1.96*std, mean+1.96*std)
    return mean, mean-1.96*std, mean+1.96*std


def get_roc_array(label0, label1):
    threpool = label0 + label1
    threpool.sort()
    label0 = np.array(label0)
    label1 = np.array(label1)
    tprlist = []
    fprlist = []
    max_th = 0
    max_tpr_fpr = 0
    TPout = 0
    FPout = 0
    TNout = 0
    FNout = 0
    sensout = 0
    accuout = 0
    f1scoreout = 0
    specout = 0
    ppvout = 0
    npvout = 0
    for threshold in threpool:
        TP = np.sum(label1 >= threshold)
        FP = np.sum(label0 >= threshold)
        TN = np.sum(label0 < threshold)
        FN = np.sum(label1 < threshold)

        spec = TN / (TN + FP)
        sens = TP / (TP + FN)

        TPR = sens
        FPR = 1 - spec
        tprlist.append(TPR)
        fprlist.append(FPR)

        if (TPR - FPR) > max_tpr_fpr:
            max_th = threshold
            max_tpr_fpr = TPR - FPR
            sensout = sens
            accuout = (TP + TN) / (TP + FN + FP + TN)
            f1scoreout = 2 * TP / (2 * TP + FP + FN)
            specout = spec
            ppvout = TP / (TP + FP)
            npvout = TN / (TN + FN)
            TPout = TP
            TNout = TN
            FPout = FP
            FNout = FN

    print('sensitivity: {}\naccuarcy: {}\n'
          'f1-score: {}\nspecificicy: {}\n'
          'ppv: {}\nnpv: {}\nthreshold: {}'.format(sensout, accuout, f1scoreout, specout,  ppvout, npvout, max_th))
    print('{}  {}\n{}  {}'.format(TPout, FPout, FNout, TNout))

    return accuout, sensout, specout, ppvout, npvout, f1scoreout

    # x = np.linspace(0, 1, 500)
    # y = np.interp(x, fprlist[::-1], tprlist[::-1])
    # return x, y
def test(model, device, test_loader):
    model.eval()
    Label0 = []
    Label1 = []
    with torch.no_grad():
        for target, cl_feature in test_loader:
            target = target.type(torch.FloatTensor)
            cl_feature = cl_feature.type(torch.FloatTensor)
            target, cl_feature = target.to(device), cl_feature.to(device)
            output = model(cl_feature)
            target = target.cpu()
            output = output.cpu().item()
            if target[0].item() == 0:
                Label0.append(output)
            else:
                Label1.append(output)
    this_auc, auc0, auc1 = get_auc_and_ci(Label0, Label1)
    acc, sen, spe, ppv, npv, f1 = get_roc_array(Label0, Label1)
    return this_auc, auc0, auc1, acc, sen, spe, ppv, npv, f1


if __name__ == '__main__':
    # 设置每次加载的图片数量
    batch_size = 16
    # 设置总的训练周期
    epoch = 20
    # 设置学习率
    learning_rate = 0.0001
    # 设置每多少步打印一次训练过程
    log_interval = 10
    # 是否保存模型
    save_model = True
    # 是否载入预训练模型
    load_pretrained = True
    # 设置硬件环境
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    label_idx = 12
    dataset_dir = '../dataset_extract/val'
    cl_feature_dir = '../../val_pool/cl_feature_array'
    root_dir = '../../val_pool/nodule_array_sysucc_wy'
    label_dict = {12: 'lvi',
                  13: 'vpi',
                  14: 'ln'}
    n_fold = 5

    auc__ = []
    auc__0 = []
    auc__1 = []
    acc__ = []
    sen__ = []
    spe__ = []
    ppv__ = []
    npv__ = []
    f1__ = []

    for fold_n in range(n_fold):
        save_model_str = 'ann_bs{}_lb{}_wabs_fold{}'.format(batch_size, label_dict[label_idx], fold_n)

        # 创建网络模型
        model = ANN(num_classes=1)
        check_point = torch.load('weights/{}.pth'.format(save_model_str))
        model.load_state_dict(check_point)

        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

        # # 加载测试集
        test_dataset = Lungdataset(root_dir=root_dir,
                                   cl_feature_dir=cl_feature_dir,
                                   dataset_dir=dataset_dir,
                                   is_train=False,
                                   label_idx=label_idx,
                                   data_subset=fold_n,
                                   is_norm=False,
                                   ct_range=(-800, 400))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

        model = model.to(device)

        # 网络测试

        print(save_model_str)
        auc, auc0, auc1, acc, sen, spe, ppv, npv, f1 = test(model, device, test_loader)
        auc__.append(auc)
        auc__0.append(auc0)
        auc__1.append(auc1)
        acc__.append(acc)
        sen__.append(sen)
        spe__.append(spe)
        ppv__.append(ppv)
        npv__.append(npv)
        f1__.append(f1)
        print('')

    print('######################### Average ##################\n'
          'auc: {} ci {} {}\n'
          'acc: {}\nsen: {}\n'
          'spe: {}\nppv: {}\n'
          'npv: {}\nf1 score: {}\n'.format(np.mean(auc__),
                                           np.mean(auc__0),
                                           np.mean(auc__1),
                                           np.mean(acc__),
                                           np.mean(sen__),
                                           np.mean(spe__),
                                           np.mean(ppv__),
                                           np.mean(npv__),
                                           np.mean(f1__)))


