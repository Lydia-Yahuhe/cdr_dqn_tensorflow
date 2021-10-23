import random

import numpy as np

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
    network = models.mlp(num_hidden=256, num_layers=4, layer_norm=True)
    if not test:
        act = deepq.learn(
            env,
            network=network,  # 隐藏节点，隐藏层数
            lr=5e-4,
            batch_size=32,
            total_timesteps=100000,
            buffer_size=100000,

            param_noise=True,
            prioritized_replay=True,
        )
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
