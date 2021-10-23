import random

import numpy as np

from baselines import deepq
from baselines.common import models
from fltenv import ConflictScene

from fltenv.env import ConflictEnv

root = ".\\dataset\\my_model"


def output_dqn_policy(env):
    data = env.train + env.test
    size = len(data)
    count = 0

    for i, info in enumerate(data):
        print(i, size, '{}%'.format(count/size*100.0))

        times = 0
        while True:
            print('\t>>> No.{} resolution,'.format(times+1), end='\t')
            scene = ConflictScene(info, limit=env.limit)
            done = False
            result = {'result': False}

            while not done:
                env_action = random.randint(0, 161)
                next_obs, rew, done, result = env.step(env_action, scene=scene)

            if result['result']:
                print('solved')
                count += 1
                break

            times += 1
            if times >= 10:
                print('failed')
                break

            print()


def train(test=False):
    env = ConflictEnv()
    network = models.mlp(num_hidden=128, num_layers=2)
    if not test:
        act = deepq.learn(
            env,
            network=network,  # 隐藏节点，隐藏层数
            lr=1e-3,
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
        output_dqn_policy(env)


if __name__ == '__main__':
    # train()
    train(test=True)
