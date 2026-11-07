from torch.utils.data import DataLoader
import argparse
import torch
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataloader import IR_Dataset
from utils.model_S2IPNet import S2IPNet


def select_gpu(device_number=0):
    if torch.cuda.device_count() >= 1:
        for i in range(torch.cuda.device_count()):
            if i == device_number and i < torch.cuda.device_count():
                return torch.device(f"cuda:{device_number}")
    return torch.device("cpu")


device = select_gpu()


def SoftIoULoss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1
    intersection = pred * target
    loss = (intersection.sum() + smooth) / (
        pred.sum() + target.sum() - intersection.sum() + smooth
    )
    return 1 - loss.mean()


def batch_pix_accuracy(output, target):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    predict = (output > 0).astype("int64")
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = (F.sigmoid(output).detach().cpu().numpy() > 0.5).astype("int64")
    target = target.cpu().numpy().astype("int64")
    intersection = predict * (predict == target)
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all()
    return area_inter, area_union


class mIoUMetric:
    def __init__(self, nclass):
        super(mIoUMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pix_acc = self.total_correct / (np.spacing(1) + self.total_label)
        iou = self.total_inter / (np.spacing(1) + self.total_union)
        return pix_acc, iou.mean()

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def para_parser():
    arg_parser = argparse.ArgumentParser(description="small target detection")
    arg_parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="number of epochs to train (default: 500)",
    )
    arg_parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        metavar="N",
        help="start epochs (default:0)",
    )
    arg_parser.add_argument("--batch_size", default=8)
    arg_parser.add_argument(
        "--in_channels", type=int, default=3, help="in_channel=1 for pre-process"
    )
    arg_parser.add_argument(
        "--need_data_normalize",
        type=bool,
        default=False,
        help="whether to normalize data True/False",
    )
    arg_parser.add_argument("--gpu_number", type=int, default=0)
    arg_parser.add_argument(
        "--dataset_root_path", type=str, default="dataset", help="dataset_root_path"
    )
    arg_parser.add_argument(
        "--dataset",
        type=str,
        default="SIRST",
        help="dataset name: NUDT-SIRST, SIRST, IRSTD-Air",
    )
    arg_parser.add_argument("--model", type=str, default="S2IPNet", help="model name")
    arg_parser.add_argument(
        "--SAM2_mode", type=str, default="s", choices=["s", "b+", "l", "t"]
    )
    arg_parser.add_argument(
        "--decay_weights",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.125, 0.0625],
    )
    arg_parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    arg_parser.add_argument(
        "--min_lr", default=1e-5, type=float, help="minimum learning rate"
    )
    return arg_parser.parse_args()


def train_head(args):
    print(f"{args.dataset} Dataset Loading ......")

    train_dataset = IR_Dataset(
        args.dataset_root_path,
        args.dataset,
        train=True,
        image_channel=args.in_channels,
        need_data_normalize=args.need_data_normalize,
    )
    test_dataset = IR_Dataset(
        args.dataset_root_path,
        args.dataset,
        test=True,
        image_channel=args.in_channels,
        need_data_normalize=args.need_data_normalize,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    print(f"{args.model} Model Initializing ......")
    img_size = train_dataset.feat_size
    model = S2IPNet(
        imgsz=img_size, in_channels=args.in_channels, SAM2_mode=args.SAM2_mode
    ).to(device)

    loss = SoftIoULoss

    return model, loss, train_dataloader, test_dataloader


def make_dir(dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    save_path = "%s_%s_%s" % (dataset, model, dt_string)
    os.makedirs(os.path.join("result", save_path), exist_ok=True)
    return save_path


def save_train_log(args, save_path):
    dict_args = vars(args)
    args_key = list(dict_args.keys())
    args_value = list(dict_args.values())
    with open(os.path.join("result", save_path, "train_log.txt"), "w") as f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
        f.write(dt_string)
        f.write("\n")
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(":--")
            f.write(str(args_value[i]))
            f.write("\n")
    return
