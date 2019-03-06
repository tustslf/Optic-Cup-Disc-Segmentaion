# -*- coding:utf-8 -*-
# author: YangZhang time:19/02/2019

from skimage import io
import Model_Disc_Det
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from utils import mk_dir, dice_coef, return_list
import torch
from torchvision import transforms
import os

DiscSeg_size = 640

# training data and label path
train_img_path = "REFUGE-Training400/Training400/Glaucoma/"
train_label_path = "Annotation-Training400/Annotation-Training400/Disc_Cup_Masks/Glaucoma/"

# validation data and label path
val_img_path = "REFUGE-Validation400/REFUGE-Validation400/"
val_label_path = "REFUGE-Validation400-GT/Disc_Cup_Masks/"

data_save_path = mk_dir("training_crop/data/")
label_save_path = mk_dir("training_crop/label/")


class MyTrainData(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, data_type, train=True):
        self.train = train
        self.data_path = data_path
        self.label_path = label_path
        if self.train:  # train the model
            self.train_list = return_list(data_path, data_type)  # the images name list of train and validation
        else:  # validate the model
            self.val_list = return_list(data_path, data_type)  # the images name list of train and validation

    def __getitem__(self, idx):
        if self.train:
            img_no = self.train_list[idx]  # the img_no looks like "g001.jpg"
            img_name = img_no.split(".")  # split the img_no by '.'
            org_img = io.imread(self.data_path + img_no)  # read the image
            org_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3))  # resize the image to a lower level
            org_img = torch.from_numpy(org_img)
            org_img = org_img.view(3, 640, 640).float()

            label_img = io.imread(self.label_path + img_name[0] + ".bmp")
            # label image has three pixel values: 0, 128, 255,
            # we would like to regard the the optic disk area as 1, other area as 0
            label_img[label_img < 200] = 1
            label_img[label_img > 200] = 0
            # as the resized image pixel value has changed, we need to multiply 255 to make pixel value equals {0,1}
            label_img = resize(label_img, (DiscSeg_size, DiscSeg_size), order=0)*255
            label_img = torch.from_numpy(label_img).float()
            return org_img, label_img
        else:
            img_no = self.val_list[idx]
            img_name = img_no.split(".")
            val_img = io.imread(self.data_path + img_no)
            val_img = resize(val_img, (DiscSeg_size, DiscSeg_size, 3))
            val_img = torch.from_numpy(val_img)
            val_img = val_img.view(3, 640, 640).float()

            val_label = io.imread(self.label_path + img_name[0] + ".bmp")
            val_label[val_label < 200] = 1
            val_label[val_label > 200] = 0
            val_label = resize(val_label, (DiscSeg_size, DiscSeg_size), order=0) * 255
            val_label = torch.from_numpy(val_label).float()
            return val_img, val_label

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.val_list)


if torch.cuda.is_available():
    DiscSeg_model = Model_Disc_Det.Net().cuda()
else:
    DiscSeg_model = Model_Disc_Det.Net()
# optimization settings
optimizer = torch.optim.SGD(DiscSeg_model.parameters(), lr=0.0001)

# train the disc detection model
train_set = MyTrainData(train_img_path, train_label_path, data_type=".jpg", train=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
train_len = train_set.__len__()
img = 0
DiscSeg_model.train()
for i, (train_img, train_label) in enumerate(train_loader):

    pred_label = DiscSeg_model(train_img)  # forward pass
    loss = dice_coef(train_label, pred_label)  # calculate the loss

    # this part is used to check the prediction label result
    pred_label = pred_label.detach().numpy()
    pred_label = np.reshape(pred_label, (DiscSeg_size, DiscSeg_size))
    plt.imshow(pred_label)
    plt.show()
    pred_label = torch.from_numpy(pred_label).float()

    optimizer.zero_grad()  # zeros the gradient buffers of all parameters
    loss.backward()  # back-propagation
    optimizer.step()  # update after calculating the gradients
    img += 1
    print('Training Step [{}/{}], Train Loss:{:.7f}'.format(img, train_len, loss))

# test the model
val_set = MyTrainData(val_img_path, val_label_path, data_type=".jpg", train=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1)
val_len = val_set.__len__()//4
val = 0
loss_val = 0
DiscSeg_model.eval()
for j, (value_img, value_label) in enumerate(train_loader):

    pred_label_val = DiscSeg_model(value_img)  # forward pass
    loss_val = dice_coef(value_label, pred_label_val)  # calculate the loss

    pred_label_val = pred_label_val.detach().numpy()
    pred_label_val = np.reshape(pred_label_val, (DiscSeg_size, DiscSeg_size))
    plt.imshow(pred_label_val)
    plt.show()
    pred_label = torch.from_numpy(pred_label_val).float()

    val += 1
    loss_avg = loss_val / val
    print('Validation Step [{}/{}], Validation Loss:{:.7f}'.format(val, val_len, loss_avg))

'''
These codes can help show the fundus images
plt.imshow(image)
plt.show()
'''
