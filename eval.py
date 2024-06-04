import os
import sys
import pickle
import argparse
import os.path as osp
import torch
import numpy as np

sys.path.append(os.getcwd())
from src.utils.general import set_random_seed
from src.config1_m import Config
from src.net1_m_1 import Parallel_Denoiser, Series_Denoiser
from src.diff1_m_1_try import DDPM
from src.eval.xyz1_m_1 import compute_stats_xyz, get_multimodal_gt, sample_xyz
from src.dataset.xyz1_m_1_try import DatasetPIE
from torchvision import transforms as A

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='PIE_20step')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--sample-num', type=int, default=10)
    parser.add_argument('--mode', type=str, default='stats')

    parser.add_argument('--frames', type=bool, default=True)
    parser.add_argument('--vel', type=bool, default=False)
    parser.add_argument('--seg', type=bool, default=True)
    parser.add_argument('--xy_dis', type=bool, default=True)

    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()
    cfg = Config(args.cfg)

    set_random_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu_id) if args.gpu_id >= 0 else 'cpu'

    prefix_len = 16
    pred_len = 30

    prefix_num = 30
    row = 1
    col = 5

    transform = A.Compose(
        [
            A.ToPILImage(),
            A.RandomPosterize(bits=2),
            A.RandomInvert(p=0.2),
            A.RandomSolarize(threshold=50.0),
            A.RandomAdjustSharpness(sharpness_factor=2),
            A.RandomAutocontrast(p=0.2),
            A.RandomEqualize(p=0.2),
            A.ColorJitter(0.5, 0.3),
            A.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset_cls = DatasetPIE
    dataset = dataset_cls('test', prefix_len, pred_len, transforms=transform)

        ## define network
    if '_parallel_' in args.cfg:
        denoiser = Parallel_Denoiser(dataset.traj_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads,
                                        prefix_len, pred_len, cfg.diff_steps)
    elif '_series_' in args.cfg:
        denoiser = Series_Denoiser(dataset.traj_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads,
                                    prefix_len, pred_len, cfg.diff_steps)

    ddpm = DDPM(args, denoiser, cfg, device).to(device)
    ddpm.load_state_dict(torch.load(osp.join(cfg.model_dir, 'jaad.pth'.format(cfg.max_epoch))))
    ddpm.eval()

    traj_gt_arr = get_multimodal_gt(dataset, prefix_len, 0.5)
    algos = [args.cfg.split('_')[2]]
    models = {algos[0]: ddpm}

    f = open('./log/[EVAL]{}.txt'.format(args.cfg), 'w')
    compute_stats_xyz(f, algos, models, dataset, traj_gt_arr, prefix_len, pred_len, args.sample_num, device)
    f.close()
