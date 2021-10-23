"""
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
"""

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels

        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, size=None, randomize=True):
        logger.log('----------mujoco_dset----------')

        trajectory_data = np.load(expert_path)
        obs = trajectory_data['obs']
        acs = trajectory_data['acs']

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        if len(obs.shape) > 2:
            self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
            self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])
        else:
            self.obs = np.vstack(obs)
            self.acs = np.vstack(acs)

        assert len(self.obs) == len(self.acs)
        if size is None:
            size = len(self.obs)

        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)

        # for behavior cloning
        train_size = int(size*train_fraction)
        self.train_set = Dset(self.obs[:train_size, :], self.acs[:train_size, :], self.randomize)
        self.val_set = Dset(self.obs[train_size:size, :], self.acs[train_size:size, :], self.randomize)

        logger.log("Total obs: {}".format(self.obs.shape))
        logger.log("Total act: {}".format(self.acs.shape))
        logger.log("Total transitions: %d/%d, %.2f%%" % (train_size, size, train_size*100.0/size))
        logger.log('-------------------------------')

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError


# if __name__ == '__main__':
#     data_set = Mujoco_Dset(".\\deterministic.trpo.Hopper.0.00.npz")

