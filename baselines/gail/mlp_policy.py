"""
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
"""
import tensorflow as tf
import gym

from baselines.common import tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype

from baselines.acktr.utils import dense


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(name, *args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, scope, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        # placeholder
        self.pdtype = make_pdtype(ac_space)
        stochastic_ph = tf.placeholder(dtype=tf.bool, shape=(), name="stochastic_ph")
        obs_ph = U.get_placeholder(dtype=tf.float64, shape=[None] + list(ob_space.shape), name="obs_ph")

        with tf.variable_scope("ob_filter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((obs_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, self.pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, self.pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = self.pdtype.pdfromflat(pdparam)

        ac = U.switch(stochastic_ph, self.pd.sample(), self.pd.mode())
        ac = tf.expand_dims(tf.argmax(ac, axis=-1), -1)
        self._act = U.function([stochastic_ph, obs_ph], [ac, self.vpred])

        # change for BC
        expert_ac_ph = tf.placeholder(dtype=tf.int64, shape=[None, 1], name='expert_ac_ph')
        loss = tf.reduce_mean(tf.square(expert_ac_ph - ac))
        var_list = self.get_trainable_variables(scope=scope)
        self.lossandgrad = U.function(inputs=[obs_ph, expert_ac_ph, stochastic_ph],
                                      outputs=[loss] + [U.flatgrad(loss, var_list)])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1, vpred1

    def get_variables(self, scope=None):
        if scope is None:
            scope = self.scope
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

    def get_trainable_variables(self, scope=None):
        if scope is None:
            scope = self.scope
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
