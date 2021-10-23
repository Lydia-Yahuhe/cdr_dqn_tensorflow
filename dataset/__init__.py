import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns


def kl_divergence(p, q):
    return scipy.stats.entropy(p, q)


def softmax(x):
    orig_shape = x.shape
    print("orig_shape", orig_shape)

    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
        print("matrix")
    else:
        # 向量
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
        print("vector")
    return x


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()


def get_kde(x, data_array, bandwidth=0.1):
    def gauss(x):
        import math
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))

    N = len(data_array)
    res = 0
    if len(data_array) == 0:
        return 0
    for i in range(len(data_array)):
        res += gauss((x - data_array[i]) / bandwidth)
    res /= (N * bandwidth)
    return res


def get_pdf_or_kde(input_array, bins):
    bandwidth = 1.05 * np.std(input_array) * (len(input_array) ** (-1 / 5))
    x_array = np.linspace(0, bins-1, num=100)
    y_array = [get_kde(x_array[i], input_array, bandwidth) for i in range(x_array.shape[0])]
    return y_array


def visual_action_distribution(bins=6):
    dqn_policy = np.load('dqn_policy_e_Bale.npz')
    gail_policy = np.load('dqn_policy_e_Bule.npz')

    dqn_acs = dqn_policy['acs']
    print(dqn_acs.shape, dqn_acs[0])
    print(np.mean(dqn_policy['rews']))
    # dqn_acs = np.argmax(dqn_acs, axis=1)
    gail_acs = gail_policy['acs']
    print(gail_acs.shape, gail_acs[0])
    print(np.mean(gail_policy['rews']))
    # gail_acs = np.argmax(gail_acs, axis=1)

    from matplotlib.font_manager import FontProperties  # 显示中文，并指定字体
    # myfont = FontProperties(fname=r'C:/Windows/Fonts/simhei.ttf', size=14)
    # sns.set(font=myfont.get_name())

    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.rcParams['figure.figsize'] = (100, 100)  # 设定图片大小
    f = plt.figure()  # 确定画布

    size = dqn_acs.shape[-1]
    for i in range(size):
        # 可视化动作使用分布
        print('action {}:'.format(i))
        x, y = dqn_acs[:, i], gail_acs[:, i]
        print(x.shape, y.shape)

        visual(i, (size+1)//2, x, y, f, bins=bins)

    x, y = np.ravel(dqn_acs), np.ravel(gail_acs)
    print(x.shape, y.shape)
    visual(size, (size + 1) // 2, x, y, f, bins=bins)

    plt.subplots_adjust(wspace=0.2, hspace=0.5)  # 调整两幅子图的间距
    plt.savefig('distribution_action.svg')
    plt.show()


def visual(i, size, x, y, f, bins):
    sns.set()  # 设置seaborn默认格式
    np.random.seed(0)  # 设置随机种子数

    # KL散度
    dqn_kde = get_pdf_or_kde(x, bins=bins)
    gail_kde = get_pdf_or_kde(y, bins=bins)
    kl = round(kl_divergence(dqn_kde, gail_kde), 3)
    print(kl)

    f.add_subplot(size, 4, i*2+1)
    plt.hist([x, y], bins=bins, range=(0, 6), density=True, align='left', label=['DQN', 'GAIL'])  # 绘制x的密度直方图

    plt.xlabel('Action index')
    plt.ylabel("Frequency")
    # plt.xticks(np.arange(-1, 6, 1))  # 设置x轴刻度值的字体大小
    plt.yticks(np.arange(0.0, 1.05, 0.1))  # 设置y轴刻度值的字体大小
    plt.title("The histogram of DQN and GAIL policy {}".format(i+1), fontsize=12)  # 设置子图标题
    plt.legend()

    f.add_subplot(size, 4, i*2+2)
    sns.distplot(x, bins=bins, hist=False, label='DQN')  # 绘制x的密度直方图
    sns.distplot(y, bins=bins, hist=False, label='GAIL')  # 绘制y的密度直方图
    plt.xlabel('Action index')
    plt.ylabel("KDE")
    # plt.xticks(np.arange(0, 52, 2))  # 设置x轴刻度值的字体大小
    # plt.yticks(np.arange(0.0, 1.0, 0.1))  # 设置y轴刻度值的字体大小
    plt.title("The similarity of DQN and GAIL policy {}, KL divergence={}".format(i+1, kl), fontsize=12)  # 设置子图标题
    plt.legend()


# 随机数生成
def random_policy_generator():
    data_dict = {'obs': (15000, 15, 12),
                 'acs': (15000, 26)}
    tmp = {}
    for key, shape in data_dict.items():
        rand_metric = np.random.random(shape)

        if key == 'acs':
            rand_metric = softmax(rand_metric)
            print(rand_metric.sum(axis=1), rand_metric.shape)

        tmp[key] = rand_metric
    np.savez('real_trajectories.npz', **tmp)


if __name__ == '__main__':
    visual_action_distribution()

