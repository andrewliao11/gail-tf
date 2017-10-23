#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from gailtf.baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
from gailtf.baselines import logger
from gailtf.baselines.ppo1.mlp_policy import MlpPolicy
from gailtf.baselines.common.mpi_fork import mpi_fork
from gailtf.baselines import bench
from gailtf.baselines.trpo_mpi import trpo_mpi

def train(args):
    import gailtf.baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(args.env_id)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    task_name = "trpo." + args.env_id.split("-")[0] + "." + ("%.2f"%args.entcoeff)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=args.num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3,
        sample_stochastic=args.sample_stochastic, task_name=task_name, save_per_iter=args.save_per_iter,
        ckpt_dir=args.checkpoint_dir, load_model_path=args.load_model_path, task=args.task)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--task', help='Choose to do which task', type=str, choices=['train', 'sample_trajectory'], default='train')
    parser.add_argument('--sample_stochastic', type=bool, default=False)
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=1)
    parser.add_argument('--entcoeff', help='entropy coefficiency', type=float, default=0)
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=1e6)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
