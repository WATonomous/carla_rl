from paddle_base import PaddleModel, PaddleSAC, PaddleAgent
from parl.utils import tensorboard, ReplayMemory
import os
import shutil


def init_agent_and_rpm(obs_dim, action_dim, cfg):
    '''
    Initialize agent and replay buffer, and load checkpoints
        for both of them if checkpoint is specified
    '''
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
        max_size=int(train_params['memory_size']), obs_dim=obs_dim, act_dim=action_dim)
    if 'checkpoint' in cfg and cfg['checkpoint'] is not None:
        agent.restore(f"{cfg['checkpoint']}.ckpt")
        rpm.load(f"{cfg['checkpoint']}.npz")
    return agent, rpm

def log_tb_metrics(steps, metrics):
        tensorboard.add_scalar(
            "eval/runtime", metrics["eval_runtime"], steps
        )
        tensorboard.add_scalar(
            "eval/episode_reward", metrics["avg_reward"], steps
        )
        tensorboard.add_scalar(
            "eval/episode_steps", metrics["avg_steps"], steps
        )
        tensorboard.add_scalar(
            "eval/episode_success_rate", metrics["avg_success"], steps
        )
        tensorboard.add_scalar(
            "eval/max_episode_progress", metrics["max_progress"], steps
        )
        tensorboard.add_scalar(
            "eval/avg_episode_progress", metrics["avg_progress"], steps
        )
        tensorboard.add_scalar(
            "eval/max_speed", metrics["max_speed"], steps
        )
        tensorboard.add_scalar(
            "eval/avg_speed", metrics["avg_speed"], steps
        )
        tensorboard.add_scalar(
            "eval/episode_collision_rate",
            metrics["avg_collision"],
            steps,
        )

def setup_logdir(cfg):
    exp_name = cfg['exp_name']
    if exp_name is None:
         return None
    outdir = cfg['output_dir']
    logdir = f"{outdir}/{exp_name}"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    return logdir
