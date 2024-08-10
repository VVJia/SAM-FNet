import os
import random
import numpy as np
import torch
import torchvision.datasets
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools import mask as coco_mask
import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res, phase):
        self.output_size = output_size
        self.low_res = low_res
        self.phase = phase

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.phase == "train":
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        x, y = image.shape[0:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        # image = (image - torch.FloatTensor(mean)) / torch.FloatTensor(std)
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'case_name': sample['case_name']}
        return sample

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = np.any(mask, axis=2)
        masks.append(mask)

    merged_mask = np.zeros((height, width), dtype=np.uint8)
    if masks:
        for mask in masks:
            merged_mask = merged_mask | mask

    return merged_mask

class COCO_dataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, split=None, transform=None):
        super(COCO_dataset, self).__init__(img_folder, ann_file)
        self.split = split
        self.transform = transform

    def __len__(self):
        return super(COCO_dataset, self).__len__()

    def __getitem__(self, idx):
        img, target = super(COCO_dataset, self).__getitem__(idx)

        # get filename
        image_info = self.coco.loadImgs(self.ids[idx])[0]
        filename = image_info['file_name']

        # generate masks
        w, h = img.size
        segmentations = [obj['segmentation'] for obj in target]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        label_value = target[0]['category_id'] + 1
        masks[masks == 1] = label_value

        img = np.array(img)

        sample = {'image': img, 'label': masks}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = os.path.splitext(filename)[0]

        return sample

class Cancer_dataset(Dataset):
    def __init__(self, data_dir, txt_dir, transform=None):
        # train or val or test
        phase = os.path.splitext(os.path.basename(txt_dir))[0]
        file_path = os.path.join(data_dir, phase)

        self.data = [os.path.join(file_path, file) for file in os.listdir(file_path)]
        self.transform = transform  # using transform in torch!
        self.sample_list = open(txt_dir).readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        data_dic = np.load(data_path)
        image, label = data_dic['image'], data_dic['label']
        name = os.path.splitext(os.path.basename(data_path))[0]
        sample = {'image': image, 'label': label, 'case_name': name}
        if self.transform:
            sample = self.transform(sample)

        return sample

