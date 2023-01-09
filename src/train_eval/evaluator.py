import pygame
import time

# Runs policy for 3 episodes by default and returns average reward
def render_evaluate_episodes(agent, env, num_episodes, max_steps):
    pygame.init()
    display = pygame.display.set_mode(
    (env.env.CAM_RES, env.env.CAM_RES),
    pygame.HWSURFACE | pygame.DOUBLEBUF)
    avg_reward = 0.
    for k in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            steps += 1
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            env.env.display(display)
            pygame.display.flip()
            time.sleep(0.1)
    avg_reward /= num_episodes
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