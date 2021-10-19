import os
import time
import torch
from PIL import Image
from model import MobileNetV1 as MobileNet
from torchvision import transforms

TEST_IMAGES_PATH = './test_images'
MODEL_PATH = "./MobileNet.pth"
CLASSES = ['forest', 'buildings', 'glacier', 'street', 'mountain', 'sea']

net = MobileNet(class_num=6).eval()
net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for name in os.listdir(TEST_IMAGES_PATH):
    start = time.time()
    path = os.path.join(TEST_IMAGES_PATH, name)
    img = Image.open(path)
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    out = net(img_tensor)
    print("Name:{}, Class:{}, time:{}".format(name, CLASSES[torch.argmax(out, 1)], time.time() - start))

print("模型转换开始")
x = torch.rand(1, 3, 150, 150)
traced_script_module = torch.jit.trace(func=net, example_inputs=x)
traced_script_module.save("MobileNet.pt")
print("模型转换结束，已保存为MobileNet.pt")
