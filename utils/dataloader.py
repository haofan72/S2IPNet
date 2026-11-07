import os
import cv2
import numpy as np
import random
import torch
from PIL import Image
import torchvision.transforms as transforms


def select(file_list, suffix=".png"):
    png_files = []
    for file_name in file_list:
        if file_name.endswith(suffix):
            png_files.append(file_name)
    return png_files


class IR_Dataset(torch.utils.data.Dataset):
    """
    IR_Dataset is a PyTorch Dataset class for loading and preprocessing infrared image datasets.
    """

    def __init__(
        self,
        root,
        dataset_name,
        train=False,
        test=False,
        train_test_ratio=[5, 5],
        make_test_mat=False,
        data_mode=None,
        need_data_normalize=False,
        image_channel=1,
    ):
        self.need_data_normalize = need_data_normalize
        self.image_channel = image_channel
        if dataset_name not in {"IRSTD-Air", "NUDT-SIRST", "SIRST"}:
            raise ValueError(
                f"Unsupported dataset {dataset_name!r}; "
                "choose IRSTD-Air, NUDT-SIRST, or SIRST."
            )
        self.sub_floder = ["images", "masks"]
        data_suffix = ".png"
        self.feat_size = 256
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.feat_size, self.feat_size)),
                transforms.ToTensor(),
            ]
        )

        if self.need_data_normalize:
            if self.image_channel == 3:
                self.Normalize_transform = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            else:
                self.Normalize_transform = transforms.Normalize(
                    mean=[101.06385040283203], std=[34.619606018066406]
                )

        self.image_dir = os.path.join(root, dataset_name, self.sub_floder[0])
        self.mask_dir = os.path.join(root, dataset_name, self.sub_floder[1])
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f"Dataset {dataset_name!r} requires image and mask directories: "
                f"{self.image_dir!r} and {self.mask_dir!r}. "
                "Place the dataset files in these directories before training."
            )
        self.data_name = select(os.listdir(self.image_dir), suffix=data_suffix)

        total_num = len(self.data_name)

        datalist = []
        for idx in range(total_num):
            image_name = os.path.join(self.image_dir, self.data_name[idx]).replace(
                "\\", "/"
            )
            mask_name = os.path.join(self.mask_dir, self.data_name[idx]).replace(
                "\\", "/"
            )
            datalist.append((image_name, mask_name))

        if data_mode == "train/test":
            self.imlist = datalist
            self.data_num = total_num
        else:
            train_txt_path = os.path.join(
                root, dataset_name, f"splite_train_{dataset_name}.txt"
            )
            test_txt_path = os.path.join(
                root, dataset_name, f"splite_test_{dataset_name}.txt"
            )
            if os.path.exists(train_txt_path) and os.path.exists(test_txt_path):
                train_list = self.get_list_from_txt(train_txt_path, datalist)
                test_list = self.get_list_from_txt(test_txt_path, datalist)
            else:
                train_list, test_list = self.split_dataset_by_ratio(
                    datalist,
                    train_ratio=(train_test_ratio[0] / sum(train_test_ratio)),
                    seed=42,
                )
                with open(train_txt_path, "w") as f_train:
                    for img_path, _ in train_list:
                        img_name = os.path.splitext(os.path.basename(img_path))[0]
                        f_train.write(f"{img_name}\n")
                with open(test_txt_path, "w") as f_test:
                    for img_path, _ in test_list:
                        img_name = os.path.splitext(os.path.basename(img_path))[0]
                        f_test.write(f"{img_name}\n")

            if train:
                self.imlist = train_list
                self.data_num = len(self.imlist)
            elif test:
                self.imlist = test_list
                self.data_num = len(self.imlist)

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, index):
        image_dir, mask_dir = self.imlist[index]
        if self.image_channel == 1:
            img = Image.open(image_dir).convert("L")
        elif self.image_channel == 3:
            img = Image.open(image_dir).convert("RGB")
        mask = Image.open(mask_dir).convert("L")

        if self.need_data_normalize:
            img_t = self.Normalize_transform(self.transform(img))
        else:
            img_t = self.transform(img)

        mask_t = self.transform(mask)

        return image_dir.split("/")[-1], img_t, mask_t

    def split_dataset_by_ratio(self, datalist, train_ratio=0.5, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        total_num = len(datalist)
        indices = list(range(total_num))
        random.shuffle(indices)
        split_point = int(total_num * train_ratio)
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        train_list = [datalist[i] for i in train_indices]
        test_list = [datalist[i] for i in test_indices]
        return train_list, test_list

    def get_list_from_txt(self, txt_path, datalist):
        with open(txt_path, "r") as f:
            names = [line.strip() for line in f.readlines()]
        result_list = []
        for name in names:
            for img_path, mask_path in datalist:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                if img_name == name:
                    result_list.append((img_path, mask_path))
                    break
        return result_list


normal_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)
