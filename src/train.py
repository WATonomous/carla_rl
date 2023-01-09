import argparse
from paddle_base import PaddleModel, PaddleSAC, PaddleAgent
from collections import deque
import numpy as np
import yaml
from parl.utils import tensorboard, ReplayMemory
#from torch_base import TorchModel, TorchSAC, TorchAgent  # Choose base wrt which deep-learning framework you are using
import time
from typing import Dict

from env.util.env_utils import ParallelEnv, LocalEnv
from train_eval.evaluator import render_evaluate_episodes, run_evaluate_episodes
from config.env_config import EnvConfig, EXP_NAME


def main(cfg: Dict):
    # env for eval
    eval_env = LocalEnv(cfg)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    # Initialize model, algorithm, agent, replay_memory
    if cfg['framework'] == 'torch':
        CarlaModel, SAC, CarlaAgent = TorchModel, TorchSAC, TorchAgent
    elif cfg['framework'] == 'paddle':
        CarlaModel, SAC, CarlaAgent = PaddleModel, PaddleSAC, PaddleAgent
    model = CarlaModel(obs_dim, action_dim)
    model_params = cfg['model']
    algorithm = SAC(model, **model_params)
    agent = CarlaAgent(algorithm)
    train_params = cfg['train']
    rpm = ReplayMemory(
        max_size=train_params['memory_size'], obs_dim=obs_dim, act_dim=action_dim)
    if cfg['checkpoint'] is not None:
        agent.restore(f"{cfg['checkpoint']}.ckpt")
        rpm.load(f"{cfg['checkpoint']}.npz")

    model_out_dir, tb_out_dir = None, None
    if cfg['exp_name'] is not None:
        model_out_dir = f"model_{cfg['exp_name']}"
        tb_out_dir = f"tensorboard_{cfg['exp_name']}"
    total_steps = 0
    last_save_steps = 0
    test_flag = 0
    st = 0.1
    ft = 0.1
    sr = 0
    success_rate_Q = deque(maxlen=10)

    render_evaluate_episodes(agent, eval_env, 100, cfg['env']['max_steps'])
    return

    if tb_out_dir is not None:
        tensorboard.logger.set_dir(tb_out_dir)

    # Parallel environments for training
    train_envs_params = EnvConfig['train_envs_params']
    env_num = EnvConfig['env_num']
    env_list = ParallelEnv(cfg['env'], cfg['xparl_addr'], train_envs_params)

    obs_list = env_list.reset()

    while total_steps < train_params['total_steps']:
        # Train episode
        if rpm.size() < train_params['warmup_steps']:
            action_list = [
                np.random.uniform(-1, 1, size=action_dim)
                for _ in range(env_num)
            ]
        else:
            action_list = [agent.sample(obs) for obs in obs_list]
        next_obs_list, reward_list, done_list, info_list = env_list.step(
            action_list)


        # Store data in replay memory
        for i in range(env_num):
            rpm.append(obs_list[i], action_list[i], reward_list[i],
                       next_obs_list[i], done_list[i])
            if(done_list[i]==True):
                st += 1
            else:
                ft += 1
                
                       
        obs_list = env_list.get_obs()
        total_steps = env_list.total_steps
        # Train agent after collecting sufficient data
        if rpm.size() >= train_params['warmup_steps']:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                train_params['batch_size'])
            print("Learning step ", total_steps)
            #agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,batch_terminal)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
            print("Learnt the step")
        if (total_steps%100 == 0):
            print("Step #: ", total_steps)
        # Save agent
        if model_out_dir is not None and total_steps > train_params['start_save_steps'] and total_steps > last_save_steps + train_params['save_step_freq']:
            agent.save(f"{model_out_dir}/step_{total_steps}.ckpt")
            rpm.save(f"{model_out_dir}/step_{total_steps}")
            last_save_steps = total_steps

        # Evaluate episode
        if (total_steps + 1) // cfg['test_every_steps'] >= test_flag:
            while (total_steps + 1) // cfg['test_every_steps'] >= test_flag:
                test_flag += 1
            eval_runtime = time.time()
            print("Running Evaluation")
            metrics = run_evaluate_episodes(agent, eval_env, train_params['eval_episodes'])  
            print("Evaluation Done")
            eval_runtime = time.time() - eval_runtime
            tensorboard.add_scalar(
                "eval/runtime", eval_runtime, total_steps
            )
            tensorboard.add_scalar(
                "eval/episode_reward", metrics["avg_reward"], total_steps
            )
            tensorboard.add_scalar(
                "eval/episode_steps", metrics["avg_steps"], total_steps
            )
            tensorboard.add_scalar(
                "eval/episode_success_rate", metrics["avg_success"], total_steps
            )
            tensorboard.add_scalar(
                "eval/max_episode_progress", metrics["max_progress"], total_steps
            )
            tensorboard.add_scalar(
                "eval/avg_episode_progress", metrics["avg_progress"], total_steps
            )
            tensorboard.add_scalar(
                "eval/max_speed", metrics["max_speed"], total_steps
            )
            tensorboard.add_scalar(
                "eval/avg_speed", metrics["avg_speed"], total_steps
            )
            tensorboard.add_scalar(
                "eval/episode_collision_rate",
                metrics["avg_collision"],
                total_steps,
            )
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default='config/config.yml',
        help='training config file')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)