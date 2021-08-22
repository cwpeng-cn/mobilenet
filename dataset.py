from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class IntelImageClassification(Dataset):

    def __init__(self, dataset_path, mode):

        self.train_path = os.path.join(dataset_path, "seg_train/seg_train")
        self.val_path = os.path.join(dataset_path, "seg_test/seg_test")
        self.classes = ['forest', 'buildings', 'glacier', 'street', 'mountain', 'sea']
        self.id2class, self.class2id = {}, {}

        for index, name in enumerate(self.classes):
            self.id2class[index] = name
            self.class2id[name] = index

        if mode == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((150, 150)),
                transforms.Pad(10),
                transforms.RandomCrop((150, 150)),
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

        self.paths, self.categories = self.gen_collection(mode)

    def gen_collection(self, mode):
        file_paths, file_categories = [], []
        folder = self.train_path if mode == 'train' else self.val_path

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


class LossWriter:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def add(self, loss_name, loss, i):
        with open(os.path.join(self.save_dir, loss_name + ".txt"), mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()
