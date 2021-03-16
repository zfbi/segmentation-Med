import os
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class Data_Load(Dataset):
    def __init__(self, image_path, label_path, transforms=None):
        self.transforms = transforms
        self.imgs_path = []
        self.labs_path = []

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = torchvision.transforms.Compose([
                #  torchvision.transforms.Resize((128,128)),
                # torchvision.transforms.CenterCrop(96),
                torchvision.transforms.RandomRotation((-10, 10)),
                # torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])

        img_files_path = [os.path.join(image_path, file) for file in os.listdir(image_path)]
        lab_files_path = [os.path.join(label_path, file) for file in os.listdir(label_path)]
        for file in img_files_path:
            for path in os.listdir(file):
                self.imgs_path.append(os.path.join(file, path))
        for file in lab_files_path:
            for path in os.listdir(file):
                self.labs_path.append(os.path.join(file, path))
        assert (len(self.labs_path) == len(self.imgs_path))
        
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, item):
        assert (str(self.imgs_path[item]).split("\\")[-2:] == str(self.labs_path[item]).split("\\")[-2:])
        # print("Load data successfully!!!")
        img = plt.imread(str(self.imgs_path[item + 10]))
        image = self.transforms(Image.fromarray(img))
        lab = plt.imread(str(self.labs_path[item + 10]))
        label = self.transforms(Image.fromarray(lab))
        return image, label

