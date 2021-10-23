"""
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
"""

import argparse
import os.path as osp
import logging
from mpi4py import MPI

import numpy as np
import gym

from baselines import bench
from baselines import logger

from baselines.gail import mlp_policy, trpo_mpi, behavior_clone
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from fltenv.env import ConflictEnv


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=4321)
    parser.add_argument('--expert_path', type=str, default='.\\dataset\\dqn_policy_e_Bale.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='.\\checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='.\\log')
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], default='train')
    # for evaluation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=128)
    parser.add_argument('--adversary_hidden_size', type=int, default=128)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Training Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--iterations', help='number of timesteps per episode', type=int, default=10000)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_iters', help='Max iteration for training BC', type=int, default=1e3)
    return parser.parse_args()


def main():
    args = argsparser()

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    env = ConflictEnv()
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)

    ob_space = env.observation_space
    ac_space = env.action_space

    def policy_fn(name, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=4)

    pi = policy_fn("pi")  # Construct network for new policy
    U.initialize()

    checkpoint_dir = osp.abspath(args.checkpoint_dir)
    task_name = args.algo + "_gail.seed_{}.BC_{}.iters_{}".format(args.seed, args.BC_iters, args.iterations)
    save_dir_pi = osp.join(checkpoint_dir, task_name)
    print(save_dir_pi)

    if args.task == 'train':
        # expert demonstrations
        dataset = Mujoco_Dset(expert_path=args.expert_path)

        # pretrain by Behavior Clone
        if args.pretrained:
            behavior_clone.learn(pi, dataset, batch_size=16, max_iters=args.BC_iters, verbose=True)

        # GAN
        reward_giver = TransitionClassifier(ob_space, ac_space, args.adversary_hidden_size,
                                            entcoeff=args.adversary_entcoeff)

        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)

        # Pi and old pi
        old_pi = policy_fn("old_pi")
        trpo_mpi.learn(env, reward_giver, dataset, rank, pi, old_pi,
                       g_step=args.g_step, d_step=args.d_step, max_iters=args.iterations,
                       entcoeff=args.policy_entcoeff, save_dir=save_dir_pi, save_per_iter=args.save_per_iter,
                       timesteps_per_batch=32, max_kl=0.01, cg_iters=10, cg_damping=0.1, gamma=0.995,
                       lam=0.97, vf_iters=5, vf_stepsize=1e-3)
    elif args.task == 'evaluate':
        U.load_variables(save_dir_pi)
        output_gail_policy(env,
                           pi,
                           stochastic_policy=args.stochastic_policy,
                           save_path='gail_policy')
    env.close()


def output_gail_policy(env, pi, stochastic_policy=False, save_path='policy'):
    obs_array = []
    act_array = []
    rew_array = []
    n_obs_array = []
    size = len(env.test)

    episode = 0
    count = 0
    while not env.test_over():
        print(episode, size)
        obs_collected = {'obs': [], 'act': [], 'rew': [], 'n_obs': []}

        obs, done = env.reset(test=True), False
        result = {'result': True}
        while not done:
            action, vpred = pi.act(stochastic_policy, obs)
            action = action[0][0]
            next_obs, rew, done, result = env.step(action)

            obs_collected['obs'].append(obs)
            obs_collected['act'].append(action)
            obs_collected['rew'].append(rew)
            obs_collected['n_obs'].append(next_obs)
            obs = next_obs

        if result['result']:
            count += 1
            obs_array.append(obs_collected['obs'])
            act_array.append(obs_collected['act'])
            rew_array.append(obs_collected['rew'])
            n_obs_array.append(obs_collected['n_obs'])
            if count >= 100:
                break

        episode += 1

    obs_array = np.array(obs_array, dtype=np.float64)
    act_array = np.array(act_array, dtype=np.float64)
    rew_array = np.array(rew_array, dtype=np.float64)
    n_obs_array = np.array(n_obs_array, dtype=np.float64)

    print('Success Rate is {}%'.format(count * 100.0 / size))
    print(obs_array.shape, act_array.shape, rew_array.shape, n_obs_array.shape)
    np.savez(save_path + '.npz', obs=obs_array, acs=act_array, rews=rew_array, n_obs=n_obs_array)


if __name__ == '__main__':
    main()
