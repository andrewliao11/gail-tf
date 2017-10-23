#!/usr/bin/env python
from gailtf.baselines.common import set_global_seeds, tf_util as U
from gailtf.baselines import bench
import os.path as osp
import gym, logging
from gailtf.baselines import logger
import ipdb

def train(args):
    from gailtf.baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=args.num_cpu).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = "ppo." + args.env_id.split("-")[0] + "." + ("%.2f"%args.entcoeff)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=args.num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=args.entcoeff,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear', ckpt_dir=args.checkpoint_dir,
            save_per_iter=args.save_per_iter, task=args.task,
            sample_stochastic=args.sample_stochastic,
            load_model_path=args.load_model_path,
            task_name=task_name
        )
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
