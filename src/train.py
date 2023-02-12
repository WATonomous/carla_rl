import argparse
import numpy as np
import yaml
from parl.utils import tensorboard
import typing as T

from env.util.env_utils import ParallelEnv, LocalEnv
from train_eval.util import init_agent_and_rpm, setup_logdir
from train_eval.trainer import train


def main(cfg: T.Dict):
    logdir = setup_logdir(cfg)
    # env for eval
    eval_env = LocalEnv(cfg)
    # env for parallel training
    env_list = ParallelEnv(cfg)
    agent, rpm = init_agent_and_rpm(eval_env.obs_dim, eval_env.action_dim, cfg)

    model_out_dir = None

    if logdir is not None:
        model_out_dir = f"{logdir}/model"
        tensorboard.logger.set_dir(f"{logdir}/tensorboard")

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