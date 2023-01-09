from .carla_env_orig import CarlaEnv
from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='env:CarlaEnv',
)

# register(
#     id='carla_mpc-v0',
#     entry_point='gym_carla.envs:CarlaEnvMPC',
# )
