import argparse
from collections import deque
import numpy as np
from parl.utils import csv_logger, logger, tensorboard, ReplayMemory
from env_utils import ParallelEnv, LocalEnv
#from torch_base import TorchModel, TorchSAC, TorchAgent  # Choose base wrt which deep-learning framework you are using
from paddle_base import PaddleModel, PaddleSAC, PaddleAgent
from env_config import EnvConfig, EXP_NAME
from parl.utils import CSVLogger
import pygame
import time

WARMUP_STEPS = 2e3
START_SAVE_STEPS = 0
SAVE_STEP_FREQ = 1e5
EVAL_EPISODES = 10
MEMORY_SIZE = int(5e5)
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.3  # determines the relative importance of entropy term against the reward
ALPHA_MIN = 0.1
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
LOAD_PATH = None
VIS_EVAL = True
# LOAD_PATH = 'model_mar27_new_cir_a/step_240192'
# LOAD_PATH = 'model_mar19_mpc/step_130169'

SAVE_DIR = f'model_{EXP_NAME}'
TENSORBOARD_DIR = None
TENSORBOARD_DIR = f'tensorboard_{EXP_NAME}'


# Runs policy for 3 episodes by default and returns average reward
def render_evaluate_episodes(agent, env, eval_episodes):
    display = pygame.display.set_mode(
    (env.env.CAM_RES, env.env.CAM_RES),
    pygame.HWSURFACE | pygame.DOUBLEBUF)
    avg_reward = 0.
    for k in range(eval_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < env._max_episode_steps:
            steps += 1
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            env.env.render(display)
            pygame.display.flip()
            time.sleep(0.1)
            print("done step")
    avg_reward /= eval_episodes
    pygame.display.quit()
    return avg_reward

def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.0
    avg_steps = 0.0
    avg_success = 0.0
    progress_per_episode = []
    max_speed_per_episode = []
    avg_speed = 0.0
    avg_collision = 0.0
    for k in range(eval_episodes):
        obs = env.reset()
        done = False
        steps = 0
        state_info = None
        max_speed = 0
        while not done and steps < env._max_episode_steps:
            steps += 1
            print("Evaluate Predicting")
            action = agent.predict(obs)
            print("Evaluate Stepping")
            obs, reward, done, state_info = env.step(action)
            print("Evaluate Going fine")
            avg_reward += reward
            avg_speed += np.linalg.norm(state_info["velocity_t"])
            if np.linalg.norm(state_info["velocity_t"]) > max_speed:
                max_speed = np.linalg.norm(state_info["velocity_t"])
        if env.env.isSuccess:
            avg_success += 1.0
        if env.env.isCollided:
            avg_collision += 1.0
        progress_per_episode.append(state_info["progress"])
        max_speed_per_episode.append(max_speed)
        avg_steps += steps
    avg_speed /= avg_steps
    avg_reward /= eval_episodes
    avg_steps /= eval_episodes
    avg_success /= eval_episodes
    avg_collision /= eval_episodes
    progress_per_episode = np.array(progress_per_episode)
    max_speed_per_episode = np.array(max_speed_per_episode)
    return {
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_success": avg_success,
        "max_progress": progress_per_episode.max(),
        "avg_progress": np.average(progress_per_episode),
        "avg_speed": avg_speed,
        "max_speed": max_speed_per_episode.max(),
        "average_max_speed": np.average(max_speed_per_episode),
        "avg_collision": avg_collision,
    }


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.set_dir('./{}_train'.format(args.env))
    pygame.init()

    csv_logger = CSVLogger('./{}_train/log.csv'.format(args.env))



    # env for eval
    eval_env_params = EnvConfig['eval_env_params']
    eval_env = LocalEnv(args.env, eval_env_params)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    # Initialize model, algorithm, agent, replay_memory
    if args.framework == 'torch':
        CarlaModel, SAC, CarlaAgent = TorchModel, TorchSAC, TorchAgent
    elif args.framework == 'paddle':
        CarlaModel, SAC, CarlaAgent = PaddleModel, PaddleSAC, PaddleAgent
    model = CarlaModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = CarlaAgent(algorithm)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)
    if LOAD_PATH:
        agent.restore(f"{LOAD_PATH}.ckpt")
        rpm.load(f"{LOAD_PATH}.npz")
    total_steps = 0
    last_save_steps = 0
    test_flag = 0
    st = 0.1
    ft = 0.1
    sr = 0
    success_rate_Q = deque(maxlen=10)

    if VIS_EVAL:
        render_evaluate_episodes(agent, eval_env, 100)
        return

    if TENSORBOARD_DIR:
        tensorboard.logger.set_dir(TENSORBOARD_DIR)

    # Parallel environments for training
    train_envs_params = EnvConfig['train_envs_params']
    env_num = EnvConfig['env_num']
    env_list = ParallelEnv(args.env, args.xparl_addr, train_envs_params)

    obs_list = env_list.reset()

    while total_steps < args.train_total_steps:
        # Train episode
        if rpm.size() < WARMUP_STEPS:
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
        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            print("Learning step ", total_steps)
            #agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,batch_terminal)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)
            print("Learnt the step")
        if (total_steps%100 == 0):
            print("Step #: ", total_steps)
        # Save agent
        if total_steps > int(START_SAVE_STEPS) and total_steps > last_save_steps + int(SAVE_STEP_FREQ):
            agent.save(f"{SAVE_DIR}/step_{total_steps}.ckpt")
            rpm.save(f"{SAVE_DIR}/step_{total_steps}")
            last_save_steps = total_steps

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            # avg_reward = render_evaluate_episodes(agent, eval_env, EVAL_EPISODES)  
            eval_runtime = time.time()
            print("Running Evaluation")
            metrics = run_evaluate_episodes(agent, eval_env, EVAL_EPISODES)  
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
            logger.info(
                'Total steps {}, Evaluation over {} episodes, Average reward: {}'
                .format(total_steps, EVAL_EPISODES, metrics["avg_reward"]))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xparl_addr",
        default='localhost:8080',
        help='xparl address for parallel training')
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument(
        '--framework',
        default='paddle',
        help='choose deep learning framework: torch or paddle')
    parser.add_argument(
        "--train_total_steps",
        default=5e6,
        type=int,
        help='max time steps to run environment')
    parser.add_argument(
        "--test_every_steps",
        default=1e4,
        type=int,
        help='the step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()