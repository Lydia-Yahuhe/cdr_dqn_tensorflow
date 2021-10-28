import csv

import numpy as np
from matplotlib import pyplot as plt, animation

from fltenv.cmd import CmdCount
from fltsim.utils import convert_with_align


def update_1(end_idx, sr_list, interval):
    plt.cla()

    x_list = [(i+1)*interval for i in range(end_idx)]
    y_list = sr_list[:end_idx]
    plt.plot(x_list, y_list)

    plt.xlabel('Episodes')
    plt.ylabel("KDE")
    plt.xticks(np.arange(interval, end_idx*interval+10, interval))  # 设置x轴刻度值的字体大小
    plt.yticks(np.arange(0.0, 1.05, 0.1))  # 设置y轴刻度值的字体大小
    plt.title("How many actions is used 0-{}".format(end_idx*interval), fontsize=12)  # 设置子图标题
    plt.grid(ls='--')


def update_2(start, many_cmd_count, bins, interval):
    plt.cla()
    size = len(many_cmd_count)

    start_idx, end_idx = start*interval, (start+1)*interval
    y_list = many_cmd_count[start_idx: end_idx]
    plt.hist(y_list, bins=bins, range=(1, bins + 1), rwidth=0.5, density=True, align='left')
    plt.xlabel('Action number')
    plt.ylabel("KDE")
    plt.xticks(np.arange(1, bins+1, 1))  # 设置x轴刻度值的字体大小
    plt.yticks(np.arange(0.0, 1.05, 0.1))  # 设置y轴刻度值的字体大小
    plt.title("How many actions is used {}-{}({})".format(start_idx, end_idx, size), fontsize=12)  # 设置子图标题


# 统计动作使用分布
def analysis_actions():
    """
    target 1. 整体解脱率的变化（每100/1000回合统计一次，点连线的形式）
    target 2. 多少个动作解脱冲突：直方图（训练过程 → gif，每1000回合统计一次）

    """
    reader = csv.reader(open('acts_analysis.csv', 'r', newline=''))

    result_count = []
    cmd_ok_count = [0, 0, 0]
    many_cmd_count = []
    for count, [result, *actions] in enumerate(reader):
        act_count = [0, 0, 0]
        for act in actions:
            act_idx = int(act)
            agent_id, idx = act_idx // CmdCount, act_idx % CmdCount
            [alt_idx, spd_idx, hdg_idx, time_idx] = convert_with_align(idx, x=3, align=4)

            act_count[0] += int(alt_idx != '1')
            act_count[1] += int(hdg_idx != '1')
            act_count[2] += int(spd_idx != '1')

        print(result, len(actions), act_count)

        if result == 'True':
            many_cmd_count.append(len(actions))

            result_count.append(1)
            cmd_ok_count = [cmd_ok_count[i] + int(count > 0) for i, count in enumerate(act_count)]
        else:
            result_count.append(0)

    print('Success Rate: {}%'.format(np.mean(result_count)*100))

    solved_count = sum(result_count)
    print('Alt is ok: {}%'.format(cmd_ok_count[0]/solved_count*100))
    print('Spd is ok: {}%'.format(cmd_ok_count[1]/solved_count*100))
    print('Hdg is ok: {}%'.format(cmd_ok_count[2]/solved_count*100))

    interval = 1000
    fig = plt.figure()

    # target 1
    size = len(result_count)
    times = size // interval + 1
    sr_list = [np.mean(result_count[i*interval:(i+1)*interval]) for i in range(times)]
    ani = animation.FuncAnimation(fig, update_1, list(range(1, times+1)),
                                  interval=100, fargs=(sr_list, interval))
    ani.save('target_1.gif', writer='Pillow', fps=60)
    # plt.show()

    # target 2
    bins = max(many_cmd_count)
    times = len(many_cmd_count) // interval + 1
    ani = animation.FuncAnimation(fig, update_2, list(range(0, times)),
                                  interval=100, fargs=(many_cmd_count, bins, interval))
    ani.save('target_2.gif', writer='Pillow', fps=60)
    # plt.show()


if __name__ == '__main__':
    analysis_actions()


