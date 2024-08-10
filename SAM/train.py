import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

from trainer import trainer_cancer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='../datasets/segment', help='root dir for data')
parser.add_argument('--output', type=str, default='./exp')
parser.add_argument('--txt_dir', type=str,
                    default='./datasets/train.txt', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
# parser.add_argument('--max_iterations', type=int,
#                     default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
# parser.add_argument('--stop_epoch', type=int,
#                     default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=128, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='./checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=200,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--dice_param', type=float, default=0.8)

parser.add_argument('--lr_exp', type=float, default=0.9, help='The learning rate decay expotential')

# acceleration choices
parser.add_argument('--tf32', action='store_true', help='If activated, use tf32 to accelerate the training process')
parser.add_argument('--compile', action='store_true', help='If activated, compile the training model for acceleration')
parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration')

args = parser.parse_args()

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

if __name__ == "__main__":
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    net = LoRA_Sam(sam, args.rank).cuda()
    if args.compile:
        net = torch.compile(net)

    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(args.output, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer_cancer(args, net, args.output, multimask_output, low_res)