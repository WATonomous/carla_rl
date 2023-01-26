import parl
import carla
import gym
import numpy as np
from parl.utils import csv_logger, logger, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper

class LocalEnv(object):
    def __init__(self, cfg):
        hostname = f"{cfg['env']['host_base']}_1"
        self.env = gym.make(cfg['env']['name'], host=hostname, cfg=cfg)
        self.env = ActionMappingWrapper(self.env)
        self.obs_dim = self.env.state_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)



class ParallelEnv(object):
    # def __init__(self, env_name, xparl_addr, train_envs_params, init_step=0):
    def __init__(self, cfg):
        parl.connect(cfg['xparl_addr'], distributed_files=['./*'])
        num_train_hosts = cfg['num_servers'] - 1
        assert num_train_hosts > 0, "To train, at least 2 servers are needed"
        train_hostnames = [f"{cfg['env']['host_base']}_{i+1}" for i in range(1,num_train_hosts+1)]
        self.env_list = [
            CarlaRemoteEnv(hostname, cfg)
            for hostname in train_hostnames
        ]
        self.env_num = len(self.env_list)
        self.episode_reward_list = [0] * self.env_num
        self.episode_steps_list = [0] * self.env_num
        self.total_steps = 0

    def reset(self):
        obs_list = [env.reset() for env in self.env_list]
        obs_list = [obs.get() for obs in obs_list]
        self.obs_list = np.array(obs_list)
        return self.obs_list

    def step(self, action_list):
        return_list = [
            self.env_list[i].step(action_list[i]) for i in range(self.env_num)
        ]
        return_list = [return_.get() for return_ in return_list]
        return_list = np.array(return_list, dtype=object)
        self.next_obs_list = return_list[:, 0]
        self.reward_list = return_list[:, 1]
        self.done_list = return_list[:, 2]
        self.info_list = return_list[:, 3]
        return self.next_obs_list, self.reward_list, self.done_list, self.info_list

    def get_obs(self):
        for i in range(self.env_num):
            self.total_steps += 1
            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += self.reward_list[i]

            self.obs_list[i] = self.next_obs_list[i]
            if self.done_list[i]:
                tensorboard.add_scalar('train/episode_reward_env{}'.format(i),
                                       self.episode_reward_list[i],
                                       self.total_steps)
                logger.info('Train env {} done, Reward: {}'.format(
                    i, self.episode_reward_list[i]))

                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0
                obs_list_i = self.env_list[i].reset()
                self.obs_list[i] = obs_list_i.get()
                self.obs_list[i] = np.array(self.obs_list[i])
        return self.obs_list
    
    def __len__(self):
        return self.env_num



@parl.remote_class(wait=False)
class CarlaRemoteEnv(object):
    def __init__(self, hostname, cfg):
        class ActionSpace(object):
            def __init__(self,
                         action_space=None,
                         low=None,
                         high=None,
                         shape=None,
                         n=None):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n

            def sample(self):
                return self.action_space.sample()

        self.env = gym.make(cfg['env']['name'], host=hostname, cfg=cfg)
        self.env = ActionMappingWrapper(self.env)
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)


