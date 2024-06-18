import time

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from src.eval.metrics import *
import pickle
import os
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score


# code from DLow
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# code from DLow
def get_multimodal_gt(dataset, prefix_len, multimodal_threshold):
    all_data = []
    data_gen = dataset.iter_generator()
    for data_generator in data_gen:
        data = data_generator[0]
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, prefix_len - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, prefix_len:, :])
    return traj_gt_arr


# code from DLow
def get_gt(data, prefix_len):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    # gt = data[..., :, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, prefix_len:, :]


# code from DLow
def get_tensor_pose(traj_np, device):
    # traj_np = traj_np[..., :, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
    traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
    traj_ts = torch.Tensor(traj_np).to(device).permute(1, 0, 2)

    return traj_ts


def get_prediction(cond, prefix, models, algo, sample_num, pred_len, pose_dim, concat_hist=True):
    prefix = prefix.repeat((1, sample_num, 1))
    cond_repeat = cond.repeat((sample_num, 1))

    pred = models[algo].sample(prefix, pred_len, pose_dim, cond_repeat)
    if concat_hist:
        pred = torch.cat([prefix, pred], dim=0)
    pred = pred.permute(1, 0, 2).contiguous().cpu().numpy()
    if pred.shape[0] > 1:
        pred = pred.reshape(-1, sample_num, pred.shape[-2], pred.shape[-1])
    else:
        pred = pred[None, ...]
    return pred


def sample_xyz(out_path, algos, models, sample_num, prefix_num, dataset, prefix_len, pred_len, device):
    data_gen = dataset.iter_generator(step=prefix_len)
    all_preds = []
    for i, data in enumerate(data_gen):
        # if len(all_preds) == prefix_num:
        #     break

        # if np.random.rand() > 0.95:  # to random random
        #     prefix = get_tensor_pose(data, device)[:prefix_len]  # T B D
        #
        #     with torch.no_grad():
        #         for algo in algos:
        #             pred = get_prediction(prefix, models, algo, sample_num, pred_len, dataset.traj_dim,
        #                                   concat_hist=True)
        #     pred = pred.reshape(1, sample_num, prefix_len + pred_len, -1, 3)
        #     pred[..., :1, :] = 0
        #     all_preds.append(pred)
        prefix = get_tensor_pose(data, device)[:prefix_len]  # T B D

        with torch.no_grad():
            for algo in algos:
                pred = get_prediction(prefix, models, algo, sample_num, pred_len, dataset.traj_dim,
                                      concat_hist=True)
        pred = pred.reshape(1, sample_num, prefix_len + pred_len, -1, 3)
        # pred[..., :1, :] = 0
        all_preds.append(pred)
    pickle.dump(np.concatenate(all_preds, axis=0), open(out_path, 'wb'))


def compute_stats_xyz(f, algos, models, dataset, traj_gt_arr, prefix_len, pred_len, sample_num, device):
    stats_func = {'Diversity': compute_diversity, 'minDE': compute_min_de, 'avgDE': compute_avg_de,
                  'stdDE': compute_std_de, 'minFDE': compute_min_fde,
                  'avgFDE': compute_avg_fde, 'stdFDE': compute_std_fde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}

    data_gen = dataset.iter_generator()
    num_samples = 0

    intention_gt = []
    intention_pred = []

    data_len = dataset.data_len

    for i, data_generator in enumerate(data_gen):
        data = data_generator[0]
        frame = data_generator[1]    # [:, :16, :]
        seg = data_generator[2]
        vel = data_generator[3]
        ped_id = data_generator[4]
        cross_state = data_generator[5]

        xy_dis = data_generator[6].reshape(1, 2, 46)

        frame = frame.reshape(frame.shape[0], frame.shape[1], -1, 3)
        cross_state = torch.from_numpy(cross_state).to(device)
        y_onehot = cross_state.float().argmax(1)

        print(f"{i}/{data_len}")

        num_samples += 1
        gt = get_gt(data, prefix_len)
        gt_multi = traj_gt_arr[i]

        prefix = get_tensor_pose(data, device)[:prefix_len]  # T B D

        os.makedirs('./pred_results/PIE_1_m_1_pred/', exist_ok=True)
        save_path = './pred_results/PIE_1_m_1_pred/pred_{}.pkl'.format(i)

        with torch.no_grad():
            for algo in algos:

                start = time.perf_counter()
                cond = models[algo].cond_model(frame, seg, vel, xy_dis, device)

                end = time.perf_counter()
                pred = get_prediction(cond, prefix, models, algo, sample_num, pred_len, dataset.traj_dim,
                                      concat_hist=False)
                print(end-start)

                y_onehot = y_onehot.long().cpu().numpy().ravel()
                in_pred = cond.softmax(1).argmax(1).long().cpu().numpy()
                intention_gt.append(y_onehot)
                intention_pred.append(in_pred)

                saver = {'raw': data[0], 'pred': pred[0], 'gt': gt[0],
                         'intention': in_pred, 'cross': y_onehot, 'ped_id': ped_id}
                pred = pred.astype(np.float64)
                for stats in stats_names:
                    val = 0
                    for pred_i in pred:
                        val += stats_func[stats](pred_i, gt, gt_multi)
                    stats_meter[stats][algo].update(val)
        print('-' * 80, file=f)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join(
                [f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats, file=f)
        f.flush()
        pickle.dump(saver, open(save_path, 'wb'))

    print('=' * 80, file=f)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        print(str_stats, file=f)
    print('=' * 80, file=f)
    f.flush()

    a = np.concatenate(intention_pred)
    b = np.concatenate(intention_gt)
    acc = accuracy_score(a, b)
    Pre = precision_score(a, b)
    Recall = recall_score(a, b)
    F1 = f1_score(a, b)
    auc = roc_auc_score(a, b)

    print('acc = {:.3f}'.format(acc * 100), file=f)
    print('pre = {:.3f}'.format(Pre * 100), file=f)
    print('recall = {:.3f}'.format(Recall * 100), file=f)
    print('F1 = {:.3f}'.format(F1 * 100), file=f)
    print('auc = {:.3f}'.format(auc * 100), file=f)
