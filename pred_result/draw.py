import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

bone_begin = [0, 1, 0, 4, 5, 7, 4, 6, 4, 5, 11, 10, 10, 12, 13]
bone_end = [1, 3, 2, 5, 7, 9, 6, 8, 10, 11, 13, 12, 11, 14, 15]

data_folder = '/media/biao/新加卷/diffusion-motion-prediction-main/pred_results/PIE_1_m_1_pred'
data_list = os.listdir(data_folder)
data_len = len(data_list)
data_list = sorted(data_list, reverse=False)

temp = 0
for data_name in data_list:
    data_path = os.path.join(data_folder, data_name)
    data = pickle.load(open(data_path, 'rb'))
    pred = data['pred']
    pred = pred[:1, :, :].reshape(30, 48).reshape(30, 16, 3)
    gt = data['gt'].reshape(30, 16, 3)
    a = data['intention'][0]
    id = data['ped_id']
    if data['intention'][0] == [0]:
        intention_gt = 'crossing'
    elif data['intention'][0] == [1]:
        intention_gt = 'non-crossing'
    if data['cross'][0] == [0]:
        intention_pred = 'crossing'
    elif data['cross'][0] == [1]:
        intention_pred = 'non-crossing'

    if intention_pred == 'crossing' and intention_gt == 'crossing':
        print('ok')

    num = 1

    for i in range(30):

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].set_title('{}:pres_{}'.format(id, intention_pred))
        axes[1].set_title('{}:gt_{}'.format(id, intention_gt))

        for j in range(15):
            x = [pred[i, bone_begin[j], 0], pred[i, bone_end[j], 0]]
            y = [pred[i, bone_begin[j], 1], pred[i, bone_end[j], 1]]
            axes[0].plot(x, y, color='blue')
            x_1 = [gt[i, bone_begin[j], 0], gt[i, bone_end[j], 0]]
            y_1 = [gt[i, bone_begin[j], 1], gt[i, bone_end[j], 1]]
            axes[1].plot(x_1, y_1, color='blue')

        # axes[0].set_xlim(-1.0, 1.0)
        # axes[1].set_xlim(-1.0, 1.0)

        # axes[0].set_ylim(-1.0, 1.0)
        # axes[1].set_ylim(-1.0, 1.0)

        # 设置坐标尺度
        # axes[0].set_xticks(np.arange(-3.0, 3.0, 0.2))
        # axes[1].set_xticks(np.arange(-3.0, 3.0, 0.2))
        # axes[0].set_yticks(np.arange(-3.0, 3.0, 0.2))
        # axes[1].set_yticks(np.arange(-3.0, 3.0, 0.2))

        # x,y尺度一样
        axes[0].set_aspect(1/25)
        axes[1].set_aspect(1/25)

        #
        axes[0].axis('off')
        axes[1].axis('off')



        os.makedirs('./viz_1_m_1/{}'.format(temp), exist_ok=True)
        plt.savefig('./viz_1_m_1/{}/{}.png'.format(temp, num))
        plt.close()
        num = num + 1
    temp = temp + 1
    print(f"{temp}/{data_len} done!")


    if temp == 1500:
        break

