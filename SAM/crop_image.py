import copy
import gzip
import os
import pickle
import sys

from PIL import Image
from tqdm import tqdm
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from scipy.ndimage.interpolation import zoom

from pathlib import Path
import cv2
from scipy.ndimage import label

from segment_anything import sam_model_registry
from sam_lora_image_encoder import LoRA_Sam

def generate_cropped_image():
    def do_crop(img, prd, crop_size):
        h, w = img.shape[:2]
        masked_img = img.copy()
        if np.max(prd) == 0:
            # 计算中心位置
            center_row = h // 2
            center_col = w // 2
            # 计算裁剪的起始和结束位置
            min_row = max(0, center_row - crop_size[0] // 2)
            min_col = max(0, center_col - crop_size[1] // 2)
            max_row = min(h, center_row + crop_size[0] // 2)
            max_col = min(w, center_col + crop_size[1] // 2)

        else:
            masked_img[prd != 255] = 0

            rows, cols = np.where(prd == 255)
            min_row, max_row, min_col, max_col = min(rows), max(rows), min(cols), max(cols)
            rect_width = max_col - min_col + 1
            rect_height = max_row - min_row + 1

            if rect_width < crop_size[0] or rect_height < crop_size[1]:
                # 计算裁剪区域的边界
                crop_min_row = max(0, min_row - max(0, (crop_size[0] - rect_height) // 2))
                crop_max_row = min(prd.shape[0], crop_min_row + max(crop_size[0], rect_height))

                crop_min_col = max(0, min_col - max(0, (crop_size[1] - rect_width) // 2))
                crop_max_col = min(prd.shape[1], crop_min_col + max(crop_size[1], rect_width))
                min_row, max_row, min_col, max_col = crop_min_row, crop_max_row, crop_min_col, crop_max_col

        # Crop the corresponding region from the original image
        cropped_img = Image.fromarray(masked_img[min_row:max_row, min_col:max_col])

        return cropped_img

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    root = r"../datasets/dataset1/global"
    classes = ['benign', 'tumor', 'normal']
    phases = ['test']
    source = root
    target = root.replace("global", "local_seg")
    input_size = 224
    crop_size = (256, 256)
    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = 1234
    random.seed()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    rank = 4
    lora_ckpt = r"./exp4/epoch_199.pth"
    ckpt = r"./checkpoints/sam_vit_b_01ec64.pth"
    sam, img_embedding_size = sam_model_registry['vit_b'](image_size=input_size,
                                                          num_classes=1,
                                                          checkpoint=ckpt,
                                                          pixel_mean=[0, 0, 0],
                                                          pixel_std=[1, 1, 1])

    net = LoRA_Sam(sam, rank).cuda()
    net.load_lora_parameters(lora_ckpt)

    net.eval()
    for phase in phases:
        for cls in classes:
            imgs = os.listdir(os.path.join(source, phase, cls))
            for img in tqdm(imgs):
                torch.cuda.empty_cache()
                img_path = os.path.join(source, phase, cls, img)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                origin_image = copy.deepcopy(image)
                x, y = image.shape[0:2]
                if x != input_size or y != input_size:
                    image = zoom(image, (input_size / x, input_size / y, 1.0), order=3)
                inputs = torch.from_numpy(image.astype(np.float32) / 255.0)
                inputs = inputs.permute(2, 0, 1)
                inputs = inputs.unsqueeze(0).cuda()
                with torch.no_grad():
                    outputs = net(inputs, False, input_size)
                    output_masks = outputs['masks']
                    out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                    prediction = out.cpu().detach().numpy()
                    if x != input_size or y != input_size:
                        prediction = zoom(prediction, (x / input_size, y / input_size), order=0)
                cropped_image = do_crop(img=origin_image.astype(np.uint8),
                                        prd=(prediction * 255).astype(np.uint8),
                                        crop_size=crop_size)
                output_path = os.path.join(target, phase, cls, img)
                if not os.path.exists(os.path.join(target, phase, cls)):
                    os.makedirs(os.path.join(target, phase, cls))
                cropped_image.save(output_path)

if __name__ == "__main__":
    generate_cropped_image()
