import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
# from model import MobileNetV1 as MobileNet
from model import MobileNetV2 as MobileNet
from dataset import IntelImageClassification
from utils import LossWriter

LR = 0.01
EPOCH = 30
BATCH_SIZE=32
DATASET_PATH = "../datasets/Intel_image_classification"
MODEL_PATH = "./MobileNetV1.pth"
LOG_PATH = "./LogV1.txt"

device=torch.device("mps") # "cpu"
# try:
#     import torch_directml
#     device=torch_directml.device()
# except:
#     print("不支持DirctML")

net = MobileNet(num_classes=6).to(device)
optimizer = Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 12, 20, 25], gamma=0.1)
criterion = CrossEntropyLoss()
train_dataset = IntelImageClassification(dataset_path=DATASET_PATH, mode="train")
val_dataset = IntelImageClassification(dataset_path=DATASET_PATH, mode="val")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
writer = LossWriter(save_path=LOG_PATH)
best_accuracy = 0

print("Model Architecture")
print(net)

def validation():
    global best_accuracy
    net.eval()
    correct_num = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
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
length = len(train_loader)

last_time=time.time()
for epoch in range(EPOCH):
    start_time=time.time()
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print("Epoch:[{}],Step:[{}/{}], Loss:{}".format(epoch, step % length, length, loss.item()))
            writer.add(loss=loss.item(), i=step)
            print("time:",time.time()-last_time)
            last_time=time.time()
        step += 1
    print("\n测试结果:")
    print("Device:{}, Time: {} sec, Loss:{}".format(device,time.time()-start_time,loss.item()))
    validation()
    scheduler.step()
    
   