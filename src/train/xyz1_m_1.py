import numpy as np
import torch
import time
from torch.nn import functional as F

def get_tensor_pose(traj_np, device):
    traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
    traj_ts = torch.Tensor(traj_np).permute(1, 0, 2).to(device) # T B D
    
    return traj_ts

def train_xyz(f, epoch, epoch_iters, model, optimizer, dataset, batch_size, prefix_len, device):
    t_s = time.time()

    generator = dataset.sampling_generator(num_samples=epoch_iters*batch_size, batch_size=batch_size)

    cnt = 0
    train_losses = 0
    intention_loss = 0
    motion_loss = 0

    tr_nsamples = [9974, 5956, 7867]
    tr_weight = torch.from_numpy(np.min(tr_nsamples) / tr_nsamples).float().cuda()
    tr_weight = tr_weight[:2]

    for traj_np in generator:
        traj = traj_np[0]
        frame = traj_np[1]  # [:, :16, :]
        seg = traj_np[2]
        vel = traj_np[3]
        cross_state = traj_np[5]

        # 距离
        xy_dis = traj_np[6]
        # xy_dis = None

        frame = frame.reshape(frame.shape[0], frame.shape[1], -1, 3)
        cross_state = torch.from_numpy(cross_state).to(device)
        y_onehot = cross_state.float()
        # y_onehot = torch.FloatTensor(cross_state.shape[0], 2).to(device).zero_()
        # y_onehot.scatter_(1, cross_state.long(), 1)

        # cond = model.cond_model(frame, seg, vel, device)
        cond = model.cond_model(frame, seg, vel, xy_dis, device)


        loss_intention = F.binary_cross_entropy_with_logits(cond, y_onehot, weight=tr_weight)

        traj_ts = get_tensor_pose(traj, device)

        prefix = traj_ts[:prefix_len] # T B D
        gt = traj_ts[prefix_len:] # T B D

        loss_motion = model.calc_loss(prefix, gt, cond)

        loss = 0.5 * loss_motion + 0.5 * loss_intention
        # loss = loss_intention

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        intention_loss += loss_intention.item()
        motion_loss += loss_motion.item()

        train_losses += loss.item()
        cnt += 1
    dt = time.time() - t_s
    train_losses /= cnt

    intention_loss /= cnt
    motion_loss /= cnt

    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.6f}'.format(x, y) for x, y in zip(['DIFF'], [train_losses])])
    print('====> Epoch: {} Time: {:.2f} {} intention: {:.6f} motion: {:.6f} lr: {:.5f}'.format(
        epoch, dt, losses_str, intention_loss, motion_loss, lr), file=f)

    f.flush()
        
    