import PIL.Image as Image
import os

import torch.utils.data as data
from torch.utils.data import DataLoader


def make_dataset(path):
    imgs = []
    img_types = ['normal', 'benign', 'tumor']
    for i in range(3):
        img_dir_path = os.path.join(path,img_types[i])
        img_name = os.listdir(img_dir_path)
        for name in img_name:
            img_path = os.path.join(img_dir_path,name)
            crop_path = img_path.replace('global','local_seg')
            imgs.append((img_path, crop_path, i, name))
    return imgs


class LPCDataset(data.Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_path, crop_path, label, name = self.imgs[index]

        img_x = Image.open(img_path)
        crop_x = Image.open(crop_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
            crop_x = self.transform(crop_x)

        return img_x, crop_x, label, 0, 1

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    from torchvision.transforms import *
    transforms_train = Compose([
        RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.1, 0.1), shear=5.729578),
        RandomHorizontalFlip(),
        ColorJitter(0.4, 0.4, 0.4, 0),
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = LPCDataset(root=r'./datasets/global/train', transform=transforms_train)
    train_dataloaders = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    case = next(iter(train_dataloaders))

    print(case)

