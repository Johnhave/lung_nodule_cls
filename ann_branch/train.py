# --coding:utf-8--
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import Lungdataset
from model import ANN
from loss_function import WeightedABSLoss
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
    return mean


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    Label0 = []
    Label1 = []
    cls_matrix = np.zeros((2, 2))
    with torch.no_grad():

        for target, cl_feature in test_loader:
            target = target.type(torch.FloatTensor)
            cl_feature = cl_feature.type(torch.FloatTensor)
            target, cl_feature = target.to(device), cl_feature.to(device)
            output = model(cl_feature)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output > 0.5  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            target = target.cpu()
            pred = pred.cpu()
            cls_matrix[int(target[0, 0]), int(pred[0, 0])] += 1
            output = output.cpu().item()
            if target[0].item() == 0:
                Label0.append(output)
            else:
                Label1.append(output)
    this_auc = get_auc_and_ci(Label0, Label1)
    print(cls_matrix)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return this_auc


def train(log_interval, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (target, cl_feature) in enumerate(train_loader):
        target = target.type(torch.FloatTensor)
        cl_feature = cl_feature.type(torch.FloatTensor)
        # 将训练数据加载至硬件
        target, cl_feature = target.to(device), cl_feature.to(device)
        # 优化器梯度清零
        optimizer.zero_grad()
        # 前向计算
        output = model(cl_feature)
        # 损失计算
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 网络参数更新
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(cl_feature), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # writer.add_scalar('Train loss', loss.item(), batch_idx + (epoch - 1) * len(train_loader))
if __name__ == '__main__':
    # 设置每次加载的图片数量
    batch_size = 16
    # 设置总的训练周期
    epoch = 5
    # 设置学习率
    learning_rate = 0.0005
    # 设置每多少步打印一次训练过程
    log_interval = 10
    # 是否保存模型
    save_model = True
    # 是否载入预训练模型
    load_pretrained = True
    # 设置硬件环境
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    label_idx = 14
    dataset_dir = '../dataset_extract'
    cl_feature_dir = '../../cl_feature_array'
    root_dir = '../../data_pool_sysucc_gy'
    label_dict = {12: 'lvi',
                  13: 'vpi',
                  14: 'ln'}
    n_fold = 5

    max_epoch_save_model_list = [0, 0, 0, 0, 0]
    for fold_n in range(n_fold):
        save_model_str = 'ann_bs{}_lb{}_wabs_fold{}'.format(batch_size, label_dict[label_idx], fold_n)


        # 创建网络模型
        model = ANN(num_classes=1)


        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

        # 加载训练集
        train_dataset = Lungdataset(root_dir=root_dir,
                                    cl_feature_dir=cl_feature_dir,
                                    dataset_dir=dataset_dir,
                                    is_train=True,
                                    label_idx=label_idx,
                                    data_subset=fold_n,
                                    is_norm=False,
                                    ct_range=(-800, 400))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
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
        # 设置优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # 设置损失函数
        criterion = WeightedABSLoss(alpha=10, device=device)

        # 网络训练和测试
        max_auc = 0
        max_epoch = 0
        for epoch in range(1, epoch + 1):
            train(log_interval, model, device, train_loader, criterion, optimizer, epoch)
            auc = test(model, device, test_loader, criterion)
            if save_model:
                if auc > max_auc:
                    torch.save(model.state_dict(), "weights/{}.pth".format(save_model_str))
                    max_auc = auc
                    max_epoch = epoch
            print('Max Auc: {}; Epoch: {}'.format(max_auc, max_epoch))
            print(save_model_str)
            max_epoch_save_model_list[fold_n] = 'Max Auc: {}; Epoch: {}; Save: {}'.format(max_auc, max_epoch,
                                                                                          save_model_str)

    for item in max_epoch_save_model_list:
        print(item)
