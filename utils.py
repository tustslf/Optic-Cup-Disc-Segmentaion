# -*- coding:utf-8 -*-
# author: YangZhang time:19/02/2019

import os
import torch

DiscSeg_size = 640
DiscCrop_size = 800
CDSeg_size = 400

train_img_path = "REFUGE-Training400/Training400/Glaucoma/"
train_label_path = "Annotation-Training400/Annotation-Training400/Disc_Cup_Masks/Glaucoma/"

val_img_path = "REFUGE-Validation400/REFUGE-Validation400/"
val_label_path = "REFUGE-Validation400-GT/Disc_Cup_Masks/"


def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


# return_list function returns name list of all images
def return_list(data_path, data_type):
    train_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    print(len(train_list))
    return train_list


# dice_coef function calculates the loss function of optimization according to the overlapping area
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true = y_true.view(1, -1)
    # print(y_true.shape)
    y_pred = y_pred.view(1, -1)
    # print(y_pred.shape)
    # print(y_true.mul(y_pred).shape)
    intersection = torch.sum(y_true.mul(y_pred))
    return -(2. * intersection + smooth)/(torch.sum(y_true) + torch.sum(y_pred) + smooth)
