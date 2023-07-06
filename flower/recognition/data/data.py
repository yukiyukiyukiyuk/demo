from glob import glob

import cv2
import numpy as np

def output_dataset_path_list(img_path, num_class=4, ratio=0.9):
    label_name = list(open("data/label.txt"))
    for i in range(len(label_name)):
        label_name[i] = label_name[i].replace("\n", "")
    tr_data_list = []
    val_data_list = []
    for i, name in enumerate(label_name):
        data_list = glob("data/images/{}/*.png".format(name))
        select_idx = np.arange(len(data_list))
        select_idx = np.random.choice(
            select_idx, int(len(data_list) * ratio), replace=False
        )
        for k, path in enumerate(data_list):
            if k in select_idx:
                tr_data_list.append([path, i])
            else:
                val_data_list.append([path, i])
        print("label name: {},   ".format(name), end="")
        print(
            "train: {}, validation: {}".format(
                len(select_idx), len(data_list) - len(select_idx)
            )
        )

    return tr_data_list, val_data_list

class MyDataset:
    def __init__(self, dataset_list, transform=None):
        self.dataset_list = dataset_list
        self.num_data = len(dataset_list)
        self.transform = transform

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        img = cv2.imread(self.dataset_list[idx][0])
        label = self.dataset_list[idx][1]

        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        img = cv2.resize(img, (224, 224))

        if self.transform:
            out_data = self.transform(img)
        return out_data, label
