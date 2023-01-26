from train_eval.evaluator import run_and_time_evaluate
from train_eval.util import log_tb_metrics
import numpy as np

def train(eval_env, train_env, agent, rpm, train_cfg, model_out_dir):
    total_steps = 0
    last_save_steps = 0
    test_flag = 0
    obs_list = train_env.reset()

    # Training loop
    while total_steps < train_cfg['total_steps']:
        # Run step
        if rpm.size() < train_cfg['warmup_steps']:
            action_list = [
                np.random.uniform(-1, 1, size=eval_env.action_dim)
                for _ in range(len(train_env))
            ]
        else:
            action_list = [agent.sample(obs) for obs in obs_list]

        next_obs_list, reward_list, done_list, info_list = train_env.step(
            action_list)

        # Store experiences in replay memory
        for i in range(len(train_env)):
            rpm.append(obs_list[i], action_list[i], reward_list[i],
                       next_obs_list[i], done_list[i])
                       
        obs_list = train_env.get_obs()
        total_steps = train_env.total_steps

        # Train agent after collecting sufficient data
        if rpm.size() >= train_cfg['warmup_steps']:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                train_cfg['batch_size'])
            print("Learning step ", total_steps)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        # Save agent
        if model_out_dir is not None and total_steps > train_cfg['start_save_steps'] and total_steps > last_save_steps + train_cfg['save_step_freq']:
            agent.save(f"{model_out_dir}/step_{total_steps}.ckpt")
            rpm.save(f"{model_out_dir}/step_{total_steps}")
            last_save_steps = total_steps

        # Evaluate agent
        if (total_steps + 1) // train_cfg['test_every_steps'] > test_flag:
            while (total_steps + 1) // train_cfg['test_every_steps'] > test_flag:
                test_flag += 1
            metrics = run_and_time_evaluate(agent, eval_env, train_cfg['eval_episodes'])
            log_tb_metrics(total_steps, metrics)