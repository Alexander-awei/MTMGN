import os
import sys
import argparse
import os.path as osp
import torch
from tqdm import tqdm
from torchvision import transforms as A

sys.path.append(os.getcwd())
from src.utils.general import set_random_seed
from src.config1_m import Config
from src.net1_m_1 import Parallel_Denoiser, Series_Denoiser
from src.diff1_m_1_try import DDPM
from src.train.xyz1_m_1 import train_xyz
from src.dataset.xyz1_m_1_try import DatasetPIE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m_xyz_series_20step_1_m_1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)

    parser.add_argument('--frames', type=bool, default=True)
    parser.add_argument('--vel', type=bool, default=False)
    parser.add_argument('--seg', type=bool, default=True)
    parser.add_argument('--xy_dis', type=bool, default=True)

    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()
    cfg = Config(args.cfg)

    set_random_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu_id) if args.gpu_id >= 0 else 'cpu'

    os.makedirs('./log', exist_ok=True)
    f = open('./log/[TRAIN]{}.txt'.format(args.cfg), 'w')

    prefix_len = 16
    pred_len = 30

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
    dataset = dataset_cls('train', prefix_len, pred_len, transforms=transform)

    ## define networkcontain_zero
    if '_parallel_' in args.cfg:  # "parallel" model in papereinops
        denoiser = Parallel_Denoiser(dataset.traj_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads,
                                    prefix_len, pred_len, cfg.diff_steps)
    elif '_series_' in args.cfg:  # "series" model in paper
        denoiser = Series_Denoiser(dataset.traj_dim, cfg.qkv_dim, cfg.num_layers, cfg.num_heads,
                                    prefix_len, pred_len, cfg.diff_steps)

    ddpm = DDPM(args, denoiser, cfg, device).to(device)

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=cfg.learning_rate)

    for epoch in tqdm(range(cfg.max_epoch)):
        train_xyz(f, epoch, cfg.epoch_iters, ddpm, optimizer, dataset, cfg.batch_size, prefix_len, device)
        if (epoch + 1) % cfg.save_epoch == 0:
            torch.save(ddpm.state_dict(), osp.join(cfg.model_dir, '{:04}.pth'.format(epoch + 1)))
        torch.save(ddpm.state_dict(), osp.join(cfg.model_dir, 'jaad.pth'))

    f.close()