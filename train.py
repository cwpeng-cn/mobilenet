import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from model import MobileNet
from dataset import IntelImageClassification, LossWriter

LR = 0.01
EPOCH = 60
DATASET_PATH = "../datasets/Intel_image_classification"
MODEL_PATH = "./MobileNet.pth"
LOG_PATH = "./Log.txt"

net = MobileNet(class_num=6).cuda()
optimizer = Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
criterion = CrossEntropyLoss()
train_dataset = IntelImageClassification(dataset_path=DATASET_PATH, mode="train")
val_dataset = IntelImageClassification(dataset_path=DATASET_PATH, mode="val")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
writer = LossWriter(save_dir=LOG_PATH)
best_accuracy = 0


def validation():
    global best_accuracy
    net.eval()
    correct_num = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.cuda()
            out = net(imgs)
            prediction = torch.argmax(out, 1).cpu()
            correct_num += sum(prediction == labels)
        net.train()
    accuracy = correct_num / val_dataset.__len__()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(net.state_dict(), MODEL_PATH)
        print("Current Best Accuracy:{}".format(accuracy))


step = 0
for epoch in range(EPOCH):
    for imgs, labels in train_loader:
        imgs = imgs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        out = net(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print("Epoch:{}, Loss:{}".format(epoch, loss.item()))
            writer.add(loss_name="MobileNet Loss", loss=loss.item(), i=step)
        step += 1
    validation()
    scheduler.step()
