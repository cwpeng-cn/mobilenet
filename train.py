from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model import MobileNet
from dataset import IntelImageClassification

LR = 0.01
dataset_path = "../datasets/Intel_image_classification"

net = MobileNet(class_num=6)
optimizer = Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
criterion = CrossEntropyLoss()
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
