import gzip
import os
import pickle
import sys
from tqdm import tqdm
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume

from segment_anything import sam_model_registry
from datasets.dataset_cancer import Cancer_dataset
from sam_lora_image_encoder import LoRA_Sam


def inference(args, multimask_output, model, test_save_path=None):
    # db_test = db_config['Dataset'](base_dir=args.volume_path, split='val')
    db_test = Cancer_dataset(data_dir=args.data_dir, txt_dir=args.txt_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_benign = 0.0
    metric_tumor = 0.0
    metric_list = 0.0
    n_b, n_t = 0, 0
    results = {}
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, results=results)
        if "benign" in case_name:
            metric_benign += np.array(metric_i)
            n_b += 1
        else:
            metric_tumor += np.array(metric_i)
            n_t += 1
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f' % (
            i_batch, case_name, metric_i[0]))
    metric_list = metric_list / len(db_test)
    metric_benign = metric_benign / n_b
    metric_tumor = metric_tumor / n_t

    logging.info('benign mean_dice %f' % (metric_benign))
    logging.info('tumor mean_dice %f' % (metric_tumor))
    logging.info('Testing performance in best val model: mean_dice : %f' % (metric_list))
    logging.info("Testing Finished!")

    test_result = os.path.join(test_save_path, "test.gz")
    with gzip.open(test_result, "wb") as f:
        pickle.dump(results, f)
    logging.info("Saving results at %s" % (test_result))

    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--data_dir', type=str,
                        default='../datasets/segment', help='root dir for data')
    parser.add_argument('--output_dir', type=str, default='./exp/output')
    parser.add_argument('--txt_dir', type=str,
                        default='./datasets/test.txt', help='list dir')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_false', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str,
                        default='./exp/epoch_199.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    #pkg = import_module(args.module)
    net = LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, net, test_save_path)
