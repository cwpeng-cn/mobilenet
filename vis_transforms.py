from torchvision import transforms
from PIL import Image
import pylab as plt

img_path = "./test_images/605.jpg"

m_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((150, 150)),
    transforms.Pad(10),
    transforms.RandomCrop((150, 150)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

img = Image.open(img_path)
img_0 = m_transforms(img)
img_1 = m_transforms(img)
img_2 = m_transforms(img)

plt.subplot(221)
plt.xticks([]), plt.yticks([])
plt.title('original')
plt.imshow(img)

plt.subplot(222)
plt.xticks([]), plt.yticks([])
plt.title('processed 1')
plt.imshow(img_0)

plt.subplot(223)
plt.xticks([]), plt.yticks([])
plt.title('processed 2')
plt.imshow(img_1)

plt.subplot(224)
plt.xticks([]), plt.yticks([])
plt.title('processed 3')
plt.imshow(img_2)

plt.show()
