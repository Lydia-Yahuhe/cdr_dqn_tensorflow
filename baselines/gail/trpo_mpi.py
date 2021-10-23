"""
Disclaimer: The trpo part highly rely on trpo_mpi at @openai/baselines
"""

import time
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.gail.statistics import stats


def traj_segment_generator(pi, env, reward_giver, horizon, stochastic):
    # Initialize state variables
    t = 0
    new = True
    ob = env.reset()
    cur_ep_ret = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float64')
    rews = np.zeros(horizon, 'float64')
    vpreds = np.zeros(horizon, 'float64')
    news = np.zeros(horizon, 'int32')
    acs = np.zeros([horizon, 1], 'int32')
    # logger.log('----------trpo_mpi_generator----------')
    # print('obs shape:', obs.shape)
    # print('true_rews shape:', true_rews.shape)
    # print('rews shape:', rews.shape)
    # print('vpreds shape:', vpreds.shape)
    # print('news shape:', news.shape)
    # print('acs shape:', acs.shape)
    # logger.log('---------------------------------------')

    while True:
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "new": news, "ac": acs,
                   "vpred": vpreds, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_true_rets": ep_true_rets}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac

        rew = reward_giver.get_reward(ob, ac)
        ob, true_rew, new, _ = env.step(ac[0][0])
        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            ob = env.reset()
            print('step:', t)
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])

    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float64')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"].reshape([-1, ])


def learn(env, reward_giver, expert_dataset, rank, pi, old_pi, g_step, d_step, entcoeff,
          save_per_iter, save_dir, timesteps_per_batch, gamma, lam, max_kl, cg_iters,
          cg_damping=1e-2, vf_stepsize=3e-4, d_step_size=3e-4, vf_iters=3, max_iters=0):

    n_workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    # Setup losses and stuff
    # Placeholder
    atarg_ph = tf.placeholder(dtype=tf.float64, shape=[None])  # Target advantage function (if applicable)
    ret_ph = tf.placeholder(dtype=tf.float64, shape=[None])  # Empirical return
    ob_ph = U.get_placeholder_cached(name="obs_ph")
    ac_ph = tf.placeholder(dtype=tf.int64, shape=[None, 1])  # Empirical return

    logger.log('----------trpo_mpi----------')
    print('obs_ph shape', ob_ph.shape)
    print('ac_ph shape', ac_ph.shape)
    print('ret_ph shape', ret_ph.shape)
    print('atarg_ph', atarg_ph.shape)

    kl_old_and_new = old_pi.pd.kl(pi.pd)
    mean_kl = tf.reduce_mean(kl_old_and_new)
    mean_entropy = tf.reduce_mean(pi.pd.entropy())
    entbonus = entcoeff * mean_entropy

    vf_err = tf.reduce_mean(tf.square(pi.vpred - ret_ph))

    ratio = tf.exp(pi.pd.logp(ac_ph) - old_pi.pd.logp(ac_ph))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg_ph)

    # 优化目标
    optimgain = surrgain + entbonus
    losses = [optimgain, mean_kl, entbonus, surrgain, mean_entropy]
    loss_names = ["optimgain", "mean_kl", "entloss", "surrgain", "entropy"]

    dist = mean_kl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]

    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vf_adam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    kl_grads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float64, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(kl_grads, tangents)])  # pylint: disable=E1111

    assign_old_eq_new = U.function([], [],
                                   updates=[tf.assign(old_v, new_v) for old_v, new_v in zipsame(old_pi.get_variables(),
                                                                                                pi.get_variables())])
    compute_losses = U.function([ob_ph, ac_ph, atarg_ph], losses)
    compute_grad = U.function([ob_ph, ac_ph, atarg_ph], losses + [U.flatgrad(optimgain, var_list)])
    # fisher vector product
    compute_fvp = U.function([flat_tangent, ob_ph, ac_ph, atarg_ph], U.flatgrad(gvp, var_list))
    compute_vf = U.function([ob_ph, ret_ph], U.flatgrad(vf_err, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            t_start_ = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - t_start_), color='magenta'))
        else:
            yield

    def all_mean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= n_workers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vf_adam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_batch, stochastic=True)

    iters_so_far = 0
    rew_buffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rew_buffer = deque(maxlen=40)

    g_loss_states = stats(loss_names)
    d_loss_states = stats(reward_giver.loss_name)
    ep_states = stats(["True_rewards", "Rewards", "Episode_length"])

    while iters_so_far <= max_iters:
        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0:
            assert save_dir is not None
            U.save_variables(save_dir, variables=pi.get_variables())

        logger.log("********** Iteration %i ************" % iters_so_far)

        def fisher_vector_product(p):
            return all_mean(compute_fvp(p, *[arr[::5] for arr in args])) + cg_damping * p

        # ------------------ Update G ------------------
        logger.log("## Optimizing Policy...")
        for _ in range(g_step):
            logger.log('\n({}/{})'.format(_+1, g_step))
            with timed("1. sample data"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            v_pred_before = seg["vpred"].reshape([-1, ])  # predicted value function before update
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"):
                pi.ob_rms.update(ob)  # update running mean/std for policy

            args = ob, ac, atarg
            assign_old_eq_new()  # set old parameter values to new parameter values

            with timed("2. compute gradients"):
                *loss_before, g = compute_grad(*args)
                loss_before = all_mean(np.array(loss_before))
                g = all_mean(g)

            if not np.allclose(g, 0):
                with timed("3. conjugate gradients"):
                    step_dir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)

                assert np.isfinite(step_dir).all()

                shs = 0.5*step_dir.dot(fisher_vector_product(step_dir))
                full_step = step_dir / np.sqrt(shs / max_kl)
                expected_improve = g.dot(full_step)
                surr_before = loss_before[0]
                step_size = 1.0
                th_before, th_new = get_flat(), None
                for _ in range(10):
                    th_new = th_before + full_step * step_size
                    set_from_flat(th_new)
                    mean_losses = surr, kl, *_ = all_mean(np.array(compute_losses(*args)))
                    improve = surr - surr_before
                    
                    logger.log("param_sumsExpected: %.3f Actual: %.3f" % (expected_improve, improve))
                    if not np.isfinite(mean_losses).all():  # loss无限大或小
                        logger.log("param_sumsGot non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("param_sumsviolated KL constraint({}/{}).".format(kl, max_kl*1.5))
                    elif improve < 0:
                        logger.log("param_sumssurrogate didn't improve.")
                    else:
                        logger.log("param_sumsStep size OK!")
                        break
                    step_size *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(th_before)
                    
                if n_workers > 1 and iters_so_far % 20 == 0:
                    param_sums = MPI.COMM_WORLD.allgather((th_new.sum(), vf_adam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, param_sums[0]) for ps in param_sums[1:])
            else:
                logger.log("Got zero gradient. not updating")

            with timed("4. value function"):
                for _ in range(vf_iters):
                    for (mb_ob, mb_ret) in dataset.iterbatches((ob.reshape([-1, ob.shape[-1]]), seg["tdlamret"]),
                                                               include_final_partial_batch=False, batch_size=128):
                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mb_ob)  # update running mean/std for policy

                        g = all_mean(compute_vf(mb_ob, mb_ret))
                        vf_adam.update(g, vf_stepsize)

        for (name, val) in zip(loss_names, mean_losses):
            logger.record_tabular('g_'+name, val)
        logger.record_tabular("ev_tdlam_before", explained_variance(v_pred_before, tdlamret))

        # ------------------ Update D ------------------
        logger.log("\n## Optimizing Discriminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))

        batch_size = timesteps_per_batch // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(batch_size)

            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"):
                ob_update = np.concatenate((ob_batch, ob_expert), 0)
                reward_giver.obs_rms.update(ob_update)

            *new_losses, g = reward_giver.compute_grad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(all_mean(g), d_step_size)
            d_losses.append(new_losses)
        mean_d_loss = np.mean(d_losses, axis=0)
        logger.log(fmt_row(13, mean_d_loss))

        for name, val in zip(reward_giver.loss_name, mean_d_loss):
            logger.record_tabular('d_'+name, val)

        # ------------------ Record all ------------------
        listoflrpairs = MPI.COMM_WORLD.allgather((seg["ep_rets"], seg["ep_true_rets"]))
        rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rew_buffer.extend(true_rets)
        rew_buffer.extend(rews)

        logger.record_tabular("EpRewMean", np.mean(rew_buffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rew_buffer))
        logger.record_tabular("ItersSoFar", iters_so_far)

        if rank == 0:
            logger.dump_tabular()
        iters_so_far += 1


def flatten_lists(list_of_lists):
    return [el for list_ in list_of_lists for el in list_]
