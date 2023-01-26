import argparse
import numpy as np
import yaml
from parl.utils import tensorboard
import time
import typing as T

from env.util.env_utils import ParallelEnv, LocalEnv
from train_eval.evaluator import run_and_time_evaluate
from train_eval.util import init_agent_and_rpm, log_tb_metrics
from train_eval.trainer import train


def main(cfg: T.Dict):
    # env for eval
    eval_env = LocalEnv(cfg)
    # env for parallel training
    env_list = ParallelEnv(cfg)
    agent, rpm = init_agent_and_rpm(eval_env.obs_dim, eval_env.action_dim, cfg)

    model_out_dir = None
    if cfg['exp_name'] is not None:
        model_out_dir = f"model_{cfg['exp_name']}"
        tensorboard.logger.set_dir(f"tensorboard_{cfg['exp_name']}_train")

    train(eval_env, env_list, agent, rpm, cfg['train'], model_out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default='config/config.yml',
        help='training config file')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)