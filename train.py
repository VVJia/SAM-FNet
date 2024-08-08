import multiprocessing
multiprocessing.set_start_method("spawn", True)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from torch import optim
from torch.utils.data import DataLoader
from dataset import LPCDataset
from model import SAM_FNet50
import time
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from torchvision import transforms
import torch.cuda.amp as amp
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import os

def classacc(predicted, label):
    acc = label.float()+predicted.float()
    acc_normal = (acc == 0).sum().float()
    acc_tumor = (acc == 4).sum().float()
    acc_temp = label.float()*predicted.float()
    acc_benign = (acc_temp == 1).sum().float()
    return acc_normal, acc_benign, acc_tumor

def cal_loss(args, criterion, m1, m2, c1, c2, outputs, features, labels, target1, target2, epoch):
    # GAN-like Loss
    if (epoch > args.gan_epoch) and args.gan_opt:
        idx = (labels == 1) | (labels == 2)
        loss_flag = torch.zeros_like(labels).float().cuda()
        loss_flag[idx] = 1
        loss1 = criterion['cs'](features['global'], features['local'], loss_flag)

        target1 = target1.unsqueeze(1)
        target2 = target2.unsqueeze(1)
        target1 = torch.cat([1.0 - target1, target1], dim=1)
        target2 = torch.cat([1.0 - target2, target2], dim=1)

        loss_be_m1 = F.binary_cross_entropy_with_logits(m1, target1, reduction='none')
        loss_be_m2 = F.binary_cross_entropy_with_logits(m2, target2, reduction='none')
        loss_be_m1 = loss_be_m1 * loss_flag.unsqueeze(1)
        loss_be_m2 = loss_be_m2 * loss_flag.unsqueeze(1)

        loss_flag_sum = loss_flag.sum()
        if loss_flag_sum == 0:
            loss_flag_sum = torch.tensor(1e-8).to(loss_flag.device)
        loss_be_m1 = loss_be_m1.sum() / loss_flag_sum
        loss_be_m2 = loss_be_m2.sum() / loss_flag_sum
        loss2 = loss_be_m1 + loss_be_m2
    else:
        loss1 = loss2 = 0

    # cross-entropy loss - global, local, and fusion
    loss3 = criterion['fc'](outputs, labels) if args.Focal else criterion['ce'](outputs, labels)
    loss4 = criterion['ce'](c1, labels)
    loss5 = criterion['ce'](c2, labels)

    # total loss
    loss = args.gan_weight * loss1 + args.gan_weight * loss2 + \
           args.fusion_weight * loss3 + \
           args.global_weight * loss4 + \
           args.local_weight * loss5

    return loss1, loss2, loss3, loss4, loss5, loss

def train_model(args, model, criterion, train_dataloaders, val_dataloaders, num_epochs, model_path, writer, writer_):
    if args.warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0005)
        
    iter_num = 0
    max_iterations = num_epochs * len(train_dataloaders)
    print("{} iterations per epoch. {} max iterations ".format(len(train_dataloaders), max_iterations))

    scaler = amp.GradScaler()
    best_val_acc = 0
    for epoch in np.arange(0, num_epochs) + 1:
        model.train()
        print("=======Epoch:{}=======".format(epoch))
        epoch_start_time = time.time()

        epoch_loss = 0
        epoch_cos_loss = 0.0
        epoch_dis_loss = 0.0
        epoch_local_loss = 0.0
        epoch_global_loss = 0.0
        epoch_fusion_loss = 0.0

        step = 0
        correct = 0.0
        correct_normal = 0.0
        correct_benign = 0.0
        correct_tumor = 0.0
        total = 0.0
        total_normal = 0.0
        total_benign = 0.0
        total_tumor = 0.0
        for idx, (input1, input2, labels, target1, target2) in tqdm(enumerate(train_dataloaders), total=len(train_dataloaders)):   # 这边一次取出一个batchsize的东西
            step += 1
            input1, input2, labels, target1, target2 = \
                input1.cuda(), input2.cuda(), labels.cuda(), target1.cuda(), target2.cuda()
            with amp.autocast():
                c1, c2, outputs, m1, m2, features = model(input1, input2, labels)
                loss1, loss2, loss3, loss4, loss5, loss = cal_loss(args, criterion, m1, m2, c1, c2,
                                                        outputs, features, labels, target1, target2, epoch)
            current_batchsize = outputs.size()[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()*current_batchsize
            if (epoch > args.gan_epoch) and args.gan_opt:
                epoch_cos_loss += loss1.item()*current_batchsize
                epoch_dis_loss += loss2.item()*current_batchsize
            epoch_global_loss += loss4.item()*current_batchsize
            epoch_local_loss += loss5.item()*current_batchsize
            epoch_fusion_loss += loss3.item()*current_batchsize

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            correct_normal_temp, correct_benign_temp, correct_tumor_temp = classacc(predicted, labels)
            correct_normal += correct_normal_temp
            correct_benign += correct_benign_temp
            correct_tumor += correct_tumor_temp
            total_normal += (labels == 0).sum().float()
            total_benign += (labels == 1).sum().float()
            total_tumor += (labels == 2).sum().float()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = args.lr * (1.0 - shift_iter / max_iterations) ** 0.965
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_cos', loss1, iter_num)
            writer.add_scalar('info/loss_dis', loss2, iter_num)
            writer.add_scalar('info/loss_ce_global', loss4, iter_num)
            writer.add_scalar('info/loss_ce_local', loss5, iter_num)
            writer.add_scalar('info/loss_ce_fusion', loss3, iter_num)

        epochmean = epoch_loss/total
        if epoch > args.gan_epoch and args.gan_opt:
            epoch_cos_mean = epoch_cos_loss/total
            epoch_dis_mean = epoch_dis_loss/total

        epoch_global_mean = epoch_global_loss/total
        epoch_local_mean = epoch_local_loss/total
        epoch_fusion_mean = epoch_fusion_loss/total
        acc_normal = correct_normal/total_normal
        acc_benign = correct_benign/total_benign
        acc_tumor = correct_tumor/total_tumor
        acc_class_mean = (acc_normal + acc_benign + acc_tumor)/3.0
        acc_mean = correct/total
        print("train_loss_mean_%d"%epoch, epochmean)
        if (epoch > args.gan_epoch) and args.gan_opt:
            print("train_cos_loss_mean_%d" % epoch, epoch_cos_mean)
            print("train_dis_loss_mean_%d" % epoch, epoch_dis_mean)

        print("train_global_loss_%d" % epoch, epoch_global_mean)
        print("train_local_loss_%d" % epoch, epoch_local_mean)
        print("train_fusion_loss_%d" % epoch, epoch_fusion_mean)
        print("train_acc_normal_mean_%d"%epoch, acc_normal)
        print("train_acc_benign_mean_%d"%epoch, acc_benign)
        print("train_acc_tumor_mean_%d"%epoch, acc_tumor)
        print("train_acc_class_mean_%d"%epoch, acc_class_mean)
        print("train_acc_mean_%d"%epoch, acc_mean)

        ## tensorboard
        writer.add_scalar("loss", epochmean, epoch)
        if epoch > args.gan_epoch and args.gan_opt:
            writer.add_scalar("cos_loss", epoch_cos_mean, epoch)
            writer.add_scalar("dis_loss", epoch_dis_mean, epoch)

        writer.add_scalar("global_loss", epoch_global_mean, epoch)
        writer.add_scalar("local_loss", epoch_local_mean, epoch)
        writer.add_scalar("fusion_loss", epoch_fusion_mean, epoch)
        writer.add_scalar("acc_normal", acc_normal, epoch)
        writer.add_scalar("acc_benign", acc_benign, epoch)
        writer.add_scalar("acc_tumor", acc_tumor, epoch)
        writer.add_scalar("acc_class_mean", acc_class_mean, epoch)
        writer.add_scalar("acc_mean", acc_mean, epoch)

        # ---------------------- validating --------------------------
        model.eval()
        with torch.no_grad():
            epoch_loss_val = 0
            epoch_glabol_loss_val = 0.0
            epoch_local_loss_val = 0.0
            epoch_cos_loss_val = 0.0
            epoch_dis_loss_val = 0.0
            epoch_fusion_loss_val = 0.0
            step_val = 0
            correct = 0.0
            correct_normal = 0.0
            correct_benign = 0.0
            correct_tumor = 0.0
            total = 0.0
            total_normal = 0.0
            total_benign = 0.0
            total_tumor = 0.0
            for idx, (input1, input2, labels, target1, target2) in tqdm(enumerate(val_dataloaders), total=len(val_dataloaders)):
                step_val += 1
                input1, input2, labels, target1, target2 = \
                    input1.cuda(), input2.cuda(), labels.cuda(), target1.cuda(), target2.cuda()
                c1, c2, outputs, m1, m2, features = model(input1, input2, labels)
                loss1, loss2, loss3, loss4, loss5, loss = cal_loss(args, criterion, m1, m2, c1, c2,
                                                     outputs, features, labels, target1, target2, epoch)
                current_batchsize = outputs.size()[0]
                epoch_loss_val += loss.item()*current_batchsize
                if (epoch > args.gan_epoch) and args.gan_opt:
                    epoch_cos_loss_val += loss1.item()*current_batchsize
                    epoch_dis_loss_val += loss2.item()*current_batchsize
                epoch_glabol_loss_val += loss4.item()*current_batchsize
                epoch_local_loss_val += loss5.item()*current_batchsize
                epoch_fusion_loss_val += loss3.item()*current_batchsize

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                correct_normal_temp, correct_benign_temp, correct_tumor_temp = classacc(predicted, labels)
                correct_normal += correct_normal_temp
                correct_benign += correct_benign_temp
                correct_tumor += correct_tumor_temp
                total_normal += (labels == 0).sum().float()
                total_benign += (labels == 1).sum().float()
                total_tumor += (labels == 2).sum().float()

            epochmean_val = epoch_loss_val/total
            if epoch > args.gan_epoch and args.gan_opt:
                epochmean_cos_val = epoch_cos_loss_val/total
                epochmean_dis_val = epoch_dis_loss_val/total

            epochmean_global_val = epoch_glabol_loss_val / total
            epochmean_local_val = epoch_local_loss_val / total
            epochmean_fusion_val = epoch_fusion_loss_val / total
            acc_normal_val = correct_normal/total_normal
            acc_benign_val = correct_benign/total_benign
            acc_tumor_val = correct_tumor/total_tumor
            acc_class_mean_val = (acc_normal_val + acc_benign_val + acc_tumor_val)/3.0
            acc_mean_val = correct/total
            print("val_loss_mean_%d"%epoch, epochmean_val)
            if (epoch > args.gan_epoch) and args.gan_opt:
                print("val_cos_loss_mean_%d" % epoch, epochmean_cos_val)
                print("val_dis_loss_mean_%d" % epoch, epochmean_dis_val)

            print("val_global_loss_mean_%d" % epoch, epochmean_global_val)
            print("val_local_loss_mean_%d" % epoch, epochmean_local_val)
            print("val_fusion_loss_mean_%d" % epoch, epochmean_fusion_val)
            print("val_acc_normal_mean_%d"%epoch, acc_normal_val)
            print("val_acc_benign_mean_%d"%epoch, acc_benign_val)
            print("val_acc_tumor_mean_%d"%epoch, acc_tumor_val)
            print("val_acc_class_mean_%d"%epoch, acc_class_mean_val)
            print("val_acc_mean_%d"%epoch, acc_mean_val)

            ## tensorboard
            writer_.add_scalar("loss", epochmean_val, epoch)
            if epoch > args.gan_epoch and args.gan_opt:
                writer_.add_scalar("cos_loss", epochmean_cos_val, epoch)
                writer_.add_scalar("dis_loss", epochmean_dis_val, epoch)

            writer_.add_scalar("global_loss", epochmean_global_val, epoch)
            writer_.add_scalar("local_loss", epochmean_local_val, epoch)
            writer_.add_scalar("fusion_loss", epochmean_fusion_val, epoch)
            writer_.add_scalar("acc_normal", acc_normal_val, epoch)
            writer_.add_scalar("acc_benign", acc_benign_val, epoch)
            writer_.add_scalar("acc_tumor", acc_tumor_val, epoch)
            writer_.add_scalar("acc_class_mean", acc_class_mean_val, epoch)
            writer_.add_scalar("acc_mean", acc_mean_val, epoch)

            if acc_mean_val*0.1 + acc_class_mean_val*0.9 > best_val_acc:
                best_val_acc = acc_mean_val*0.1 + acc_class_mean_val * 0.9
                torch.save(model.module.state_dict(), model_path / '{}_{}.pth'.format(epoch, best_val_acc))

            print("%2.2f sec(s)"%(time.time() - epoch_start_time))

        torch.cuda.empty_cache()
        if (epoch % 5 == 0 or epoch == num_epochs - 1):  # 每隔5轮存一下
            torch.save(model.module.state_dict(), model_path / '{}.pth'.format(epoch))

    writer.close()
    writer_.close()

    return model

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='./datasets/dataset1/global/train')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=60)

    parser.add_argument('--gan_opt', action='store_true')
    parser.add_argument('--gan_epoch', type=int, default=10)
    parser.add_argument('--gan_weight', type=float, default=0.01)

    parser.add_argument('--AdamW', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmup_period', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.003)

    parser.add_argument('--global_weight', type=float, default=1.0)
    parser.add_argument('--local_weight', type=float, default=0.3)
    parser.add_argument('--fusion_weight', type=float, default=1.0)

    parser.add_argument('--save_path', type=str, default='./model_ours')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--devices', type=str, default="0, 1")

    args = parser.parse_args()

    save = Path(args.save_path)
    save.mkdir(parents=True, exist_ok=True)
    config_file = save / 'config.txt'
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    f.close()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    same_seeds(args.seed)
    model = SAM_FNet50(num_classes=3, num_features=2)
    model = DataParallel(model)
    model.cuda()
    criterion = {}
    criterion['ce'] = nn.CrossEntropyLoss()
    if args.gan_opt:
        criterion['cs'] = nn.CosineEmbeddingLoss()
        criterion['be'] = nn.BCEWithLogitsLoss(reduction='none')
    transforms_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.1, 0.1), shear=5.729578),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0),
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_val = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = LPCDataset(root = args.data_dir, transform = transforms_train)
    print('The length of training dataset:', len(train_dataset))
    train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = LPCDataset(root = args.data_dir.replace('train', 'val'), transform = transforms_val)
    print('The length of validating dataset:', len(val_dataset))
    val_dataloaders = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    record_path = save / 'runs'
    model_path = save / 'weights'
    record_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter((record_path / 'train').as_posix())
    writer_ = SummaryWriter((record_path / 'val').as_posix())
    train_model(args, model, criterion, train_dataloaders, val_dataloaders, args.epoch, model_path, writer, writer_)