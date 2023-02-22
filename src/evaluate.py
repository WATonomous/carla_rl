import argparse
from env.util.env_utils import LocalEnv
from parl.utils import tensorboard
import yaml
import typing as T
from train_eval.util import init_agent_and_rpm, log_tb_metrics, setup_logdir
from train_eval.evaluator import render_evaluate_episodes, run_and_time_evaluate

def main(cfg: T.Dict):
    logdir = setup_logdir(cfg)

    eval_env = LocalEnv(cfg)
    agent, _ = init_agent_and_rpm(eval_env.obs_dim, eval_env.action_dim, cfg)
    episodes = cfg['train']['eval_episodes']

    if cfg['env']['render']:
        render_evaluate_episodes(agent, eval_env, episodes)
    else:
        assert(logdir is not None, "Provide an experiment name to save to")
        tensorboard.logger.set_dir(f"{logdir}/tensorboard")
        metrics = run_and_time_evaluate(agent, eval_env, episodes)
        log_tb_metrics(0, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default='config/config.yml',
        help='training config file')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    main(cfg)
