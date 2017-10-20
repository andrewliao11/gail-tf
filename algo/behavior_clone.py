import tensorflow as tf
import baselines.common.tf_util as U
from baselines import logger
from tqdm import tqdm
from baselines.common.mpi_adam import MpiAdam
import tempfile
import ipdb

def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
           adam_epsilon=1e-5, optim_stepsize=3e-4, ckpt_dir=None, log_dir=None):
  ob_space = env.observation_space
  ac_space = env.action_space
  pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
  # placeholder
  ob = U.get_placeholder_cached(name="ob")
  ac = pi.pdtype.sample_placeholder([None])
  stochastic = U.get_placeholder_cached(name="stochastic")
  # use maximum liklihood
  loss = -tf.reduce_mean(pi.pd.logp(ac))
  # use mean square error
  #loss = tf.reduce_mean(tf.square(ac-pi.ac))
  var_list = pi.get_trainable_variables()
  adam = MpiAdam(var_list, epsilon=adam_epsilon)
  lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

  U.initialize()
  adam.sync()
  logger.log("Training with Behavior Cloning...")
  for iter_so_far in tqdm(range(int(max_iters))):
    ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size)
    loss, g = lossandgrad(ob_expert, ac_expert, True)
    adam.update(g, optim_stepsize)
  # save checkpoint in a temporary file
  savedir_fname = tempfile.TemporaryDirectory().name
  U.save_state(savedir_fname, var_list=pi.get_variables())
  return savedir_fname
