from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from model import MobileNet
from dataset import IntelImageClassification

LR = 0.01
EPOCH = 120
dataset_path = "../datasets/Intel_image_classification"

net = MobileNet(class_num=6).cuda()
optimizer = Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 100], gamma=0.1)
criterion = CrossEntropyLoss()
train_dataset = IntelImageClassification(dataset_path=dataset_path, is_train=True)
test_dataset = IntelImageClassification(dataset_path=dataset_path, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

for epoch in range(EPOCH):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        out = net(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print(loss.item())
    scheduler.step()
