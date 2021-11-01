import numpy as np
import cv2


def cv2_demo_trackbar():
    def nothing(x):
        pass

    # 创建一个黑色的图像，一个窗口
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # 创建颜色变化的轨迹栏
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)
    # 为 ON/OFF 功能创建开关
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while 1:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # 得到四条轨迹的当前位置
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')
        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]
    cv2.destroyAllWindows()


def cv2_tracks_bar():
    # 图像膨胀函数
    def img_dilated(img, d):
        # 定义 kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
        # 图像膨胀
        dilated = cv2.dilate(img, kernel)
        # 返回膨胀图片
        return dilated

    # 回调函数，因为只能传一个参数，不方便，所以pass
    def nothing(pos):
        pass

    # 读取图片
    img = cv2.imread("wuhan_base.jpg", 1)
    # 创建老窗口
    cv2.namedWindow('OldImg')
    # 绑定老窗口和滑动条（滑动条的数值）
    cv2.createTrackbar('D', 'OldImg', 1, 30, nothing)
    while True:
        # 提取滑动条的数值d
        d = cv2.getTrackbarPos('D', 'OldImg')
        # 滑动条数字传入函数img_dilated中，并且调用函数img_dilated
        dilated = img_dilated(img, d)
        # 绑定 img 和 dilated
        result = np.hstack([img, dilated])
        cv2.imshow('OldImg', result)
        # 设置推出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 关闭窗口
    cv2.destroyAllWindows()


# cv2_tracks_bar()


from baselines import deepq
from baselines.common import models

from fltenv.env import ConflictEnv

root = ".\\dataset\\my_model"


def output_dqn_policy(act, env, save_path='policy'):
    obs_array = []
    act_array = []
    rew_array = []
    n_obs_array = []
    size = len(env.test)
    episode = 1
    count = 0
    while not env.test_over():
        print(episode, size)
        obs_collected = {'obs': [], 'act': [], 'rew': [], 'n_obs': []}

        obs, done = env.reset(test=True), False
        result = {'result': True}

        while not done:
            action = act(np.array(obs)[None])[0]
            env_action = action
            # env_action = random.randint(0, 53)
            next_obs, rew, done, result = env.step(env_action)

            obs_collected['obs'].append(obs)
            obs_collected['act'].append(env_action)
            obs_collected['rew'].append(rew)
            obs_collected['n_obs'].append(next_obs)
            obs = next_obs

        if result['result']:
            count += 1
            # obs_array.append(obs_collected['obs'])
            # act_array.append(obs_collected['act'])
            # rew_array.append(obs_collected['rew'])
            # n_obs_array.append(obs_collected['n_obs'])

            obs_array += obs_collected['obs']
            act_array += obs_collected['act']
            rew_array += obs_collected['rew']
            n_obs_array += obs_collected['n_obs']

        episode += 1

    obs_array = np.array(obs_array, dtype=np.float64)
    act_array = np.array(act_array, dtype=np.float64)
    rew_array = np.array(rew_array, dtype=np.float64)
    n_obs_array = np.array(n_obs_array, dtype=np.float64)

    print('Success Rate is {}%'.format(count * 100.0 / size))
    print(obs_array.shape, act_array.shape, rew_array.shape, n_obs_array.shape)
    np.savez(save_path+'.npz', obs=obs_array, acs=act_array, rews=rew_array, n_obs=n_obs_array)


def train(test=False):
    env = ConflictEnv(limit=0)
    network = models.mlp(num_hidden=256, num_layers=2)
    if not test:
        act = deepq.learn(
            env,
            network=network,  # 隐藏节点，隐藏层数
            lr=5e-4,
            batch_size=32,
            total_timesteps=200000,
            buffer_size=100000,

            param_noise=True,
            prioritized_replay=True,
        )
        env.close()
        print('Save model to my_model.pkl')
        act.save(root+'.pkl')
    else:
        act = deepq.learn(env,
                          network=network,
                          total_timesteps=0,
                          load_path=root+".pkl")
        output_dqn_policy(act, env, save_path='dqn_policy')


if __name__ == '__main__':
    train()
    # train(test=True)
