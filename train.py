import os
import torch
from model import MobileNet
import torchvision
import pylab as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import IntelImageClassification
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

LR = 0.01

net = MobileNet(class_num=6)

optimizer = Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
criterion = CrossEntropyLoss()

dataset_path = "../datasets/Intel_image_classification"
train_dataset = IntelImageClassification(dataset_path=dataset_path, is_train=True)
test_dataset = IntelImageClassification(dataset_path=dataset_path, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

for imgs, labels in train_loader:
    optimizer.zero_grad()
    out = net(imgs)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    print(loss)
