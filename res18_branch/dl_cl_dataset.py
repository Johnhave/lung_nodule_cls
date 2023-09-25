from torch.utils.data import Dataset
import os
import torch
import numpy as np


class Lungdataset(Dataset):
    def __init__(self, root_dir, cl_feature_dir, dataset_dir, is_train, label_idx, data_subset=0, is_norm=True,
                 ct_range=(-800, 400),
                 is_patient_id=False):
        self.root_dir = root_dir
        self.is_train = is_train
        self.cl_feature_dir = cl_feature_dir
        self.is_norm = is_norm
        self.ct_range = ct_range
        self.dataset_dir = dataset_dir
        self.data_subset = data_subset
        self.label_idx = label_idx
        self.is_patient_id = is_patient_id

        self.image_file_list, self.label = self.load_dataset()

    def __getitem__(self, index):
        img_file0 = self.image_file_list[index]
        cl_feature_file = '{}.npy'.format('_'.join(img_file0.split('_')[:2]))
        img_file = os.path.join(self.root_dir, img_file0)
        label = self.label[index]
        image = np.load(img_file)
        if self.is_norm:
            image = (image - image.mean()) / image.std()
        else:
            image[image < self.ct_range[0]] = self.ct_range[0]
            image[image > self.ct_range[1]] = self.ct_range[1]
            image = (image - self.ct_range[0]) / (self.ct_range[1] - self.ct_range[0])

        cl_feature = np.load(os.path.join(self.cl_feature_dir, cl_feature_file))
        cl_feature = np.squeeze(cl_feature)
        Size0, Size1, Size2 = image.shape
        data = np.pad(image, ((0, 64 - Size0), (0, 64 - Size1), (0, 64 - Size2)), constant_values=0)
        data = torch.from_numpy(data).unsqueeze(0).float()

        # return data, np.array([label])

        if self.is_patient_id:
            return data, np.array([label]), cl_feature, self.image_file_list[index].split('_')[0]
        else:
            return data, np.array([label]), cl_feature

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

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    dataset = Lungdataset(is_train=False, is_norm=True)

    image, label = dataset[50]

    print(image.shape, label)
