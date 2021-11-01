import csv

import numpy as np
from matplotlib import pyplot as plt, animation

from fltsim.utils import convert_with_align
from script.tools_4 import update_1, update_2


def check_which_type_8(act_count):
    alt_check = act_count[0] > 0
    hdg_check = act_count[1] > 0
    spd_check = act_count[2] > 0

    if alt_check:
        if hdg_check:
            if spd_check:
                return 7
            else:
                return 6  # all have
        else:
            if spd_check:
                return 5
            else:
                return 1  # all have
    else:
        if hdg_check:
            if spd_check:
                return 4
            else:
                return 2  # all have
        else:
            if spd_check:
                return 3
            else:
                return 8  # all have


# 统计动作使用分布(3个解脱时机(0,15,30)，航向、高度、速度调整(3*3*3)，共81个动作)
def analysis_actions():
    """
    target 1. 整体解脱率的变化（每100/1000回合统计一次，点连线的形式）
    target 2. 多少个动作解脱冲突：直方图（训练过程 → gif，每1000回合统计一次）

    """
    reader = csv.reader(open('acts_analysis_1.csv', 'r', newline=''))
    CmdCount = 81

    result_count = []
    many_cmd_count = []

    statistic_type = {}
    for count, [result, *actions] in enumerate(reader):
        # if count < 20000:
        #     continue

        act_count = [0, 0, 0]
        for act in actions:
            act_idx = int(act)
            agent_id, idx = act_idx // CmdCount, act_idx % CmdCount
            [alt_idx, spd_idx, hdg_idx, time_idx] = convert_with_align(idx, x=3, align=4)

            act_count[0] += int(alt_idx != '1')
            act_count[1] += int(hdg_idx != '1')
            act_count[2] += int(spd_idx != '1')

        # print(result, len(actions), act_count)
        type_ = check_which_type_8(act_count)
        if type_ in statistic_type.keys():
            statistic_type[type_].append([int(result == 'True'), len(actions)])
        else:
            statistic_type[type_] = [[int(result == 'True'), len(actions)]]

        if result == 'True':
            many_cmd_count.append(len(actions))
            result_count.append(1)
        else:
            result_count.append(0)

    episodes = len(result_count)
    print('\nSuccess Rate: {}%'.format(np.mean(result_count) * 100))
    solved_count = sum(result_count)

    print('\nFailed Rate: {}%'.format((1-np.mean(result_count)) * 100))
    failed_count = episodes - solved_count

    x_list, y_list_1_1, y_list_2_1 = [], [], []
    y_list_1_2, y_list_1_3, y_list_2_2,  y_list_2_3 = [], [], [], []

    for key in sorted(statistic_type):
        type_list = np.array(statistic_type[key])

        result_type_list = type_list[:, 0]
        x_list.append(key)
        y_list_1_1.append(np.mean(result_type_list))
        y_list_2_1.append(len(type_list))

        y_list_1_2.append(sum(result_type_list)/solved_count)
        y_list_1_3.append((len(type_list)-sum(result_type_list))/failed_count)

        solved, failed = [], []
        for i, ele in enumerate(type_list[:, 1]):
            if type_list[i, 0] == 0:
                failed.append(ele)
            else:
                solved.append(ele)

        y_list_2_2.append(np.mean(solved)*1000)
        y_list_2_3.append(np.mean(failed)*1000)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(x_list, y_list_1_1, 'g', label='sr in each type')
    ax1.plot(x_list, y_list_1_2, 'b', label='the ratio of each type in solved scenarios')
    ax1.plot(x_list, y_list_1_3, 'y', label='the ratio of each type in failed scenarios')

    ax1.set_yticks(np.arange(0.0, 1.1, 0.1))  # 设置y轴刻度值的字体大小
    ax1.set_ylabel('Ratio')
    ax1.set_title("Total: {} Solved: {}, Failed: {}".format(len(result_count), solved_count, failed_count))
    plt.legend()

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x_list, y_list_2_1, 'r', label='the ratio of each type in total scenarios')
    ax2.plot(x_list, y_list_2_2, 'g+', label='the mean number (1000) of solved action of each type')
    ax2.plot(x_list, y_list_2_3, 'rx', label='the mean number (1000) of failed action of each type')

    ax2.set_ylabel('Number')
    ax2.set_xlabel('Same X for both exp(-x) and ln(x)')

    plt.legend()
    plt.grid(ls='--')
    plt.show()

    interval = 1000
    fig = plt.figure()

    # target 1
    size = len(result_count)
    times = size // interval + 1
    sr_list = [np.mean(result_count[i * interval:(i + 1) * interval]) for i in range(times)]
    ani = animation.FuncAnimation(fig, update_1, list(range(1, times + 1)),
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
