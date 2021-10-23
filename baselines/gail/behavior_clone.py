"""
The code is used to train BC imitator, or pretrained GAIL imitator
"""
from tqdm import tqdm

from baselines import logger

from baselines.common import tf_util as U
from baselines.common.mpi_adam import MpiAdam


def learn(pi, dataset, batch_size=128, max_iters=1e4, adam_epsilon=1e-5, lr=3e-4, verbose=False):
    assert max_iters >= 0

    adam = MpiAdam(pi.get_trainable_variables(), epsilon=adam_epsilon)

    U.initialize()
    adam.sync()

    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(batch_size, 'train')

        train_loss, g = pi.lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, lr)

        if verbose and iter_so_far % 100 == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = pi.lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))
    logger.log("Pretraining is over!")
