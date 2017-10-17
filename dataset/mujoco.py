import sys
sys.path.append("/home/andrewliao11/gail-tf")
from baselines import logger
import pickle as pkl
import numpy as np
from tqdm import tqdm
import ipdb

class Mujoco_Dset(object):
    def __init__(self, expert_path, ret_threshold=None, traj_limitation=np.inf, random=True):
        with open(expert_path, "rb") as f:
            traj_data = pkl.load(f)
        obs = []
        acs = []
        rets = []
        lens = []
        for traj in tqdm(traj_data):
            if ret_threshold is not None and traj["ep_ret"] < ret_threshold:
                pass
            if len(rets) >= traj_limitation:
                break
            rets.append(traj["ep_ret"])
            lens.append(len(traj["ob"]))
            obs.append(traj["ob"])
            acs.append(traj["ac"])
        self.num_traj = len(rets)
        self.avg_ret = sum(rets)/len(rets)
        self.avg_len = sum(lens)/len(lens)
        self.rets = np.array(rets)
        self.lens = np.array(lens)
        self.obs = np.array([v for ob in obs for v in ob])
        self.acs = np.array([v for ac in acs for v in ac])
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_transition = len(self.obs)
        self.randomize = random
        self.init_pointer()
        self.log_info()

    def log_info(self):
        logger.log("Total trajectories: %d"%self.num_traj)
        logger.log("Total transitions: %d"%self.num_transition)
        logger.log("Average episode length: %f"%self.avg_len)
        logger.log("Average returns: %f"%self.avg_ret)

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_transition)
            np.random.shuffle(idx)
            self.obs = self.obs[idx, :]
            self.acs = self.acs[idx, :]

    def get_next_batch(self, batch_size):
        if self.pointer + batch_size >= self.num_transition:
            self.init_pointer()
        end = self.pointer + batch_size
        obs = self.obs[self.pointer:end, :]
        acs = self.acs[self.pointer:end, :]
        self.pointer = end
        return obs, acs

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path):
    dset = Mujoco_Dset(expert_path)
    dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../baselines/ppo1/ppo.Hopper.0.00.pkl")
    args = parser.parse_args()
    test(args.expert_path)

