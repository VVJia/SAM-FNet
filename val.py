import os
import argparse
import multiprocessing
multiprocessing.set_start_method("spawn", True)

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import SAM_FNet50, SAM_FNet18, SAM_FNet34
from torchvision import transforms
from dataset import LPCDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

import pandas as pd

classes = {
    0: 'normal',
    1: 'benign',
    2: 'tumor'
}

def count_metrics(plist, tlist, save_path):
    pred_np = np.array(plist)
    targets_np = np.array(tlist)

    report = classification_report(targets_np, pred_np, digits=4)
    print(report)

    # Save the classification report string to a file
    with open(save_path / 'classification_report.txt', 'w') as file:
        file.write(report)

    file.close()


def count_pred(data):
    data_soft = F.softmax(data, dim=1)
    _, predicted = torch.max(data_soft.data, 1)
    return data_soft, predicted


def test(args, model, val_dataloaders, save_path):
    model.eval()
    preds = []
    targets = []
    output_scores_list = []

    with torch.no_grad():
        for idx, (input1, input2, labels, _, _) in tqdm(enumerate(val_dataloaders), total=len(val_dataloaders)):
            input1, input2, labels = input1.cuda(), input2.cuda(), labels.cuda()
            o_g, o_l, o_f, _, _, _ = model(input1, input2, labels)

            if args.ensemble:
                output = (o_g + o_l + o_f) / 3.0
            else:
                output = o_f

            output, predicted = count_pred(output)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            output_scores_list.extend(output.cpu().numpy())

        preds = [classes[x] for x in preds]
        targets = [classes[x] for x in targets]
        results = pd.DataFrame({'preds': preds, 'targets': targets})
        for i in range(3):
            results[f'class_{i}_score'] = [output_scores[i] for output_scores in output_scores_list]
        results.to_csv(save_path / "results.csv", index=False, header=True)

        count_metrics(preds, targets, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model_ours/weights/46_0.9646.pth')
    parser.add_argument('--encoder', type=str, default='ResNet50', help="encoder name",
                        choices=['ResNet18', 'ResNet34', 'ResNet50'])
    parser.add_argument('--dataset', type=str, default='dataset1')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--ensemble', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='./model_ours/')
    parser.add_argument('--devices', type=str, default='0,1')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    model_path = args.model_path
    save_path = Path(args.save_path)
    save_path = save_path / args.dataset
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    if args.encoder == 'ResNet18':
        model = SAM_FNet18(num_classes=args.num_classes, num_features=2, pretrained=False)
    elif args.encoder == 'ResNet34':
        model = SAM_FNet34(num_classes=args.num_classes, num_features=2, pretrained=False)
    elif args.encoder == 'ResNet50':
        model = SAM_FNet50(num_classes=args.num_classes, num_features=2, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    transforms_val = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = LPCDataset(root=f'./datasets/{args.dataset}/global/test', transform=transforms_val)
    print('The length of testing dataset', len(test_dataset))
    test_dataloaders = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test(args, model, test_dataloaders, save_path)
