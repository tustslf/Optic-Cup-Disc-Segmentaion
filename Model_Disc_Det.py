# -*- coding:utf-8 -*-
# author: YangZhang time:15/02/2019

from torch import autograd, nn, cat, mean
import torch.nn.functional as F
import cv2


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # conv is convolution layer
        self.bn1 = nn.BatchNorm2d(32)  # bn is batch normalization
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # deconv is deconvolution layer

        self.conv11 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv13 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.conv14 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv15 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.conv16 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn16 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.conv17 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn17 = nn.BatchNorm2d(32)
        self.conv18 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn18 = nn.BatchNorm2d(32)
        self.conv19 = nn.Conv2d(32, 1, 3, padding=1)
        self.bn19 = nn.BatchNorm2d(1)

        self.conv20 = nn.Conv2d(1, 1, 1)

    def forward(self, input_img):
        conv_1 = F.relu(self.bn1(self.conv1(input_img)))
        conv_1 = F.relu(self.bn2(self.conv2(conv_1)))
        pool_1 = F.max_pool2d(conv_1, 2)

        conv_2 = F.relu(self.bn3(self.conv3(pool_1)))
        conv_2 = F.relu(self.bn4(self.conv4(conv_2)))
        pool_2 = F.max_pool2d(conv_2, 2)

        conv_3 = F.relu(self.bn5(self.conv5(pool_2)))
        conv_3 = F.relu(self.bn6(self.conv6(conv_3)))
        pool_3 = F.max_pool2d(conv_3, 2)

        conv_4 = F.relu(self.bn7(self.conv7(pool_3)))
        conv_4 = F.relu(self.bn8(self.conv8(conv_4)))
        pool_4 = F.max_pool2d(conv_4, 2)

        conv_5 = F.relu(self.bn9(self.conv9(pool_4)))
        conv_5 = F.relu(self.bn10(self.conv10(conv_5)))

        conv_6 = F.relu(self.bn11(self.conv11(cat((conv_4, self.deconv1(conv_5)), 1))))
        conv_6 = F.relu(self.bn12(self.conv12(conv_6)))
        conv_7 = F.relu(self.bn13(self.conv13(cat((conv_3, self.deconv2(conv_6)), 1))))
        conv_7 = F.relu(self.bn14(self.conv14(conv_7)))
        conv_8 = F.relu(self.bn15(self.conv15(cat((conv_2, self.deconv3(conv_7)), 1))))
        conv_8 = F.relu(self.bn16(self.conv16(conv_8)))
        conv_9 = F.relu(self.bn17(self.conv17(cat((conv_1, self.deconv4(conv_8)), 1))))
        conv_9 = F.relu(self.bn18(self.conv18(conv_9)))
        output = F.relu(self.bn19(self.conv19(conv_9)))
        output = F.sigmoid(self.conv20(output))

        return output
