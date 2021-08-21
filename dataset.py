from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class IntelImageClassification(Dataset):
    def __init__(self, dataset_path, is_train=True):
        self.train_path = os.path.join(dataset_path, "seg_train/seg_train")
        self.test_path = os.path.join(dataset_path, "seg_test/seg_test")
        self.classes = ['forest', 'buildings', 'glacier', 'street', 'mountain', 'sea']
        self.id2class, self.class2id = {}, {}
        for index, name in enumerate(self.classes):
            self.id2class[index] = name
            self.class2id[name] = index

        if is_train:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((224, 224)),
                transforms.Pad(10),
                transforms.RandomCrop((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.paths, self.categories = self.gen_collection(is_train)

    def gen_collection(self, is_train):
        file_paths, file_categories = [], []
        folder = self.train_path if is_train else self.test_path

        for name in self.classes:
            dir_folder = os.path.join(folder, name)
            for file_name in os.listdir(dir_folder):
                file_path = os.path.join(dir_folder, file_name)
                file_paths.append(file_path)
                file_categories.append(name)

        return file_paths, file_categories

    def __getitem__(self, item):
        image = Image.open(self.paths[item])
        data = self.transforms(image)
        category = self.categories[item]
        label = self.class2id[category]
        return data, label

    def __len__(self):
        return len(self.paths)
