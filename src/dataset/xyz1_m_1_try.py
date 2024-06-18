import copy
import math
import pickle
import numpy as np
import os

import torch


class Dataset:
    def __init__(self, mode, t_his, t_pred):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.prepare_data()
        self.std, self.mean = None, None
        self.data_len = len(self.key_list)
        self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        self.normalized = False
        self.sample_ind = None

    def prepare_data(self):
        raise NotImplementedError

    def normalize_data(self, mean=None, std=None):
        if mean is None:
            all_seq = []
            for data_s in self.data.values():
                for seq in data_s.values():
                    all_seq.append(seq[:, 1:])
            all_seq = np.concatenate(all_seq)
            self.mean = all_seq.mean(axis=0)
            self.std = all_seq.std(axis=0)
        else:
            self.mean = mean
            self.std = std
        for data_s in self.data.values():
            for action in data_s.keys():
                data_s[action][:, 1:] = (data_s[action][:, 1:] - self.mean) / self.std
        self.normalized = True

    def get_mean_std(self):
        all_seq = []
        for data_s in self.data.values():
            for seq in data_s.values():
                all_seq.append(seq[:, 1:])
        all_seq = np.concatenate(all_seq)
        self.mean = all_seq.mean(axis=0)
        self.std = all_seq.std(axis=0)

    def sample(self):
        ped_id_choice = np.random.choice(list(self.key_list))
        seq = self.data[ped_id_choice]
        traj_np = seq[0]
        kp = seq[1]
        m = seq[2]
        vel = seq[3]
        ped_id = seq[4]
        cross = seq[5]
        xy = seq[6]
        return traj_np, kp, m, vel, ped_id, cross, xy

    def sampling_generator(self, num_samples=1000, batch_size=8):
        for i in range(num_samples // batch_size):
            sample = []
            kp_sample = []
            map_sample = []
            vel_sample = []
            id_sample = []
            cross_state = []
            xy = []
            for j in range(batch_size):
                sample_i, kp_i, map_i, vel_i, id_i, cross_i, xy_i = self.sample()
                sample.append(sample_i)
                kp_sample.append(kp_i)
                map_sample.append(map_i)
                vel_sample.append(vel_i)
                id_sample.append(id_i)
                cross_state.append(cross_i)
                xy.append(xy_i)
            sample = np.concatenate(sample, axis=0)
            kp_sample = np.concatenate(kp_sample, axis=0)
            map_sample = np.concatenate(map_sample, axis=0)
            vel_sample = np.concatenate(vel_sample, axis=0)
            cross_state = np.concatenate(cross_state, axis=0)
            yield sample, kp_sample, map_sample, vel_sample, id_sample, cross_state, xy

    def iter_generator(self):
        for ped_id, data_s in self.data.items():
            traj = data_s[0]
            kp = data_s[1]
            map = data_s[2]
            vel = data_s[3]
            cross = data_s[5]

            xy = data_s[6]

            yield traj, kp, map, vel, ped_id, cross, xy


class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        self._parents = np.array([])

        self._joints_right = [2, 4, 6, 8, 10, 12, 14, 16]

        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        self.skeleton_end = [0, 1, 2, 5, 6, 7, 8, 11, 12, 13, 14]
        for i in self.skeleton_end:
            self._has_children[i] = True

        self._children = []
        self._children = [[1, 2, 5, 6, 11, 12], [3], [], [4], [], [7], [9], [], [8], [10], [], [13], [15], [], [14],
                          [16], []]


class DatasetPIE(Dataset):
    def __init__(self, mode, t_his=25, t_pred=100, use_vel=False, transforms=None):
        self.use_vel = use_vel
        self.transforms = transforms
        super().__init__(mode, t_his, t_pred)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.data_file = '/media/biao/新加卷/dataset with seg_map/fze'
        self.subjects_split = {'train': 'train.pkl',
                               'test': 'test_0-10000.pkl'}
        self.subjects = self.subjects_split[self.mode]
        self.skeleton = Skeleton(
            parents=[0, 1, 3, 0, 2, 4, 0, 6, 8, 10, 0, 5, 7, 9, 0, 11, 13, 15, 0, 12, 14, 16],
            joints_left=[1, 3, 5, 7, 9, 11, 13, 15],
            joints_right=[2, 4, 6, 8, 10, 12, 14, 16])

        self.kept_joints = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.process_data()

    def process_data(self):
        data_s = {}

        with open(os.path.join(self.data_file, self.subjects), 'rb') as file:
            data_set = pickle.load(file)

        self.key_list = list(data_set.keys())

        for id, value in data_set.items():
            data_list = []
            seq = copy.deepcopy(value['kps_norm'])
            seq = seq.reshape(seq.shape[0], -1, 3)
            seq[:, 1:] -= seq[:, :1]
            seq = seq.reshape(1, 46, 17, 3)
            data_list.append(seq)
            data_list.append(value['kps'].reshape(1, 46, 51))

            if value['crossing'] == 1:
                cross_state = np.array([[1, 0]])
            else:
                cross_state = np.array([[0, 1]])
            # seg_map = torch.from_numpy(value['map'][:1])
            img = torch.from_numpy(value['map'])
            # img = torch.from_numpy(value['map'][1:])
            # img = self.transforms(img.permute(1, 2, 0)).contiguous()
            #
            # img = torch.cat([seg_map, img], 0)
            img = img.reshape(1, 4, 192, 64)

            # vel_obd = np.asarray(value['obd_speed']).reshape(1, -1) / 120.0  # normalize
            # vel_gps = np.asarray(value['gps_speed']).reshape(1, -1) / 120.0  # normalize
            # vel = torch.from_numpy(np.concatenate([vel_gps, vel_obd], 0)).float().contiguous()
            # vel = vel.reshape(1, 2, 46)
            vel = np.ones((1, 2, 46))
            data_list.append(img)
            data_list.append(vel)
            data_list.append(id)
            data_list.append(cross_state)

            # 距离
            person_cen = value['center']
            traffic = value['traffic_bbox']

            xy_l_1 = []
            xy_l_2 = []

            for t1, f1 in enumerate(traffic):
                f1 = f1[:3]
                person_1 = np.round(person_cen[t1], decimals=3)
                for t2, f2 in enumerate(f1):
                    xy_d = np.round(np.array([(f2[0] + f2[2]) / 2, (f2[1] + f2[3]) / 2]), decimals=3)
                    distance = np.round(
                        math.sqrt(((person_1[0] - xy_d[0]) / 1920) ** 2 + ((person_1[1] - xy_d[1]) / 1080) ** 2),
                        decimals=3)
                    if t2 == 0:
                        xy_l_1.append(distance)
                    elif t2 == 1:
                        xy_l_2.append(distance)

            xy_l_1 = np.array(xy_l_1)
            xy_l_2 = np.array(xy_l_2)

            if len(xy_l_1) < 46 or len(xy_l_2) < 46:
                while xy_l_2.shape != (46,):
                    xy_l_2 = np.append(xy_l_2, 0)
                while xy_l_1.shape != (46,):
                    xy_l1 = np.append(xy_l_1, 0)

            xy_l_t = []
            xy_l_t.append(xy_l_1.reshape(1, 46))
            xy_l_t.append(xy_l_2.reshape(1, 46))
            xy_l_t = np.concatenate(xy_l_t, axis=0)

            # xy_l_t = np.ones((2, 46))

            data_list.append(xy_l_t)

            data_s[id] = data_list

        self.data = data_s


def readCSVasFloat(filename):
    returnArray = []

    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]).reshape((17, 3)))

    returnArray = np.array(returnArray)

    # with open(file_path, 'r') as f:
    #     data_str = f.read()
    #
    # data = np.array(data_str)

    return returnArray
