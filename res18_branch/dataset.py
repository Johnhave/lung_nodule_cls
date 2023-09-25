from torch.utils.data import Dataset
import os
import torch
import numpy as np

class Lungdataset(Dataset):
    def __init__(self, root_dir, dataset_dir, is_train, label_idx, data_subset=0, is_norm=True, ct_range=(-800, 400),
                 is_patient_id=False):
        self.root_dir = root_dir
        self.is_train = is_train
        self.is_norm = is_norm
        self.ct_range = ct_range
        self.dataset_dir = dataset_dir
        self.data_subset = data_subset
        self.label_idx = label_idx
        self.is_patient_id = is_patient_id

        self.image_file_list, self.label = self.load_dataset()

    def __getitem__(self, index):

        img_file = os.path.join(self.root_dir, self.image_file_list[index])
        label = self.label[index]
        image = np.load(img_file)
        if self.is_norm:
            image = (image - image.mean()) / image.std()
        else:
            image[image < self.ct_range[0]] = self.ct_range[0]
            image[image > self.ct_range[1]] = self.ct_range[1]
            image = (image - self.ct_range[0]) / (self.ct_range[1] - self.ct_range[0])
        Size0, Size1, Size2 = image.shape
        data = np.pad(image, ((0, 64 - Size0), (0, 64 - Size1), (0, 64 - Size2)), constant_values=0)
        data = torch.from_numpy(data).unsqueeze(0).float()

        # return data, np.array([label])

        if self.is_patient_id:
            return data, np.array([label]), self.image_file_list[index].split('_')[0]
        else:
            return data, np.array([label])

    def __len__(self):
        return len(self.label)

    def load_dataset(self):
        image_file_list = []
        label_list = []
        if not self.is_train:
            data_txt = open(os.path.join(self.dataset_dir,
                                         str(self.label_idx),
                                         'data_subset_{}.txt'.format(self.data_subset)),
                            'r', encoding='gbk')
            for line in data_txt.readlines():
                image_file, label_ = line.rstrip().split(',')
                image_file_list.append(image_file)
                label_list.append(int(label_))
            data_txt.close()

        else:
            for txt_file in os.listdir(os.path.join(self.dataset_dir, str(self.label_idx))):
                if txt_file[-5] == str(self.data_subset):
                    continue
                data_txt = open(os.path.join(os.path.join(self.dataset_dir, str(self.label_idx)), txt_file),
                                'r', encoding='gbk')
                for line in data_txt.readlines():
                    image_file, label_ = line.rstrip().split(',')
                    image_file_list.append(image_file)
                    label_list.append(int(label_))
                data_txt.close()
        return image_file_list, label_list

if __name__ == '__main__':
    label_idx = 14
    dataset_dir = '../dataset_extract'
    root_dir = '../../data_pool_sysucc_gy'
    fold_n = 0
    dataset = Lungdataset(root_dir=root_dir, dataset_dir=dataset_dir, is_train=False, label_idx=label_idx,
                                    data_subset=fold_n, is_norm=True, ct_range=(-800, 400))


    print(len(dataset))