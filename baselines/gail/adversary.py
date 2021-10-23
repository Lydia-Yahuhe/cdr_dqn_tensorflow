"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import tensorflow as tf
import numpy as np
from gym.spaces import Discrete

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U


def logsigmoid(a):
    """Equivalent to tf.log(tf.sigmoid(a))"""
    return -tf.nn.softplus(-a)


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class TransitionClassifier(object):
    def __init__(self, ob_space, ac_space, hidden_size, entcoeff=0.001, scope="adversary"):
        print('----------adversary_classifier----------')
        self.scope = scope
        self.observation_shape = tuple(ob_space.shape)
        self.actions_shape = (ac_space.n, ) if isinstance(ac_space, Discrete) else tuple(ac_space.shape)
        print('observation_shape:', self.observation_shape, self.actions_shape)
        print('actions_shape:', self.actions_shape)

        # Build placeholder
        self.generator_obs_ph = tf.placeholder(tf.float64, (None,) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float64, (None,) + self.actions_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float64, (None,) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float64, (None,) + self.actions_shape, name="expert_actions_ph")
        print('generator_obs_ph:', self.generator_obs_ph.shape)
        print('generator_acs_ph', self.generator_acs_ph.shape)
        print('expert_obs_ph', self.expert_obs_ph.shape)
        print('expert_acs_ph', self.expert_acs_ph.shape)

        # Build graph
        generator_logits = self.__build_graph(self.generator_obs_ph, self.generator_acs_ph,
                                              hidden_size=hidden_size, reuse=False)
        expert_logits = self.__build_graph(self.expert_obs_ph, self.expert_acs_ph,
                                           hidden_size=hidden_size, reuse=True)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))

        # Build regression loss
        # let x = logits, z = targets. z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits,
                                                              labels=tf.ones_like(expert_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.reduce_mean(expert_loss)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy

        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss

        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        var_list = self.get_trainable_variables()
        self.compute_grad = U.function(
            inputs=[self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            outputs=self.losses + [U.flatgrad(self.total_loss, var_list)]
        )
        print('----------------------------------------')

    def __build_graph(self, obs_ph, acs_ph, hidden_size, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)

            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h = tf.contrib.layers.fully_connected(_input, hidden_size*8, activation_fn=tf.nn.tanh)
            p_h = tf.contrib.layers.fully_connected(p_h, hidden_size*4, activation_fn=tf.nn.tanh)
            p_h = tf.contrib.layers.fully_connected(p_h, hidden_size*2, activation_fn=tf.nn.tanh)
            p_h = tf.contrib.layers.fully_connected(p_h, hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        print(acs.shape)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)

        acs = tf.one_hot(acs)
        print(acs.shape)
        sess = tf.get_default_session()
        reward = sess.run(self.reward_op, feed_dict={self.generator_obs_ph: obs, self.generator_acs_ph: acs})
        return reward[0][0]

