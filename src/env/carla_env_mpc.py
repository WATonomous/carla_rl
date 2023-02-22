#!/usr/bin/env python

# This file is modified by Dongjie yu (yudongjie.moon@foxmail.com)
# from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# author: Jianyu Chen (jianyuchen@berkeley.edu)

from __future__ import division
import copy
import numpy as np
from typing import Dict

import carla
from math import pi

from .util.mpc import mpc_control

from .carla_env_orig import CarlaEnv


class CarlaEnvMPC(CarlaEnv):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, host: str, cfg: Dict):
        super(CarlaEnvMPC, self).__init__(host, cfg)
        self.mpc = mpc_control()

    def reset(self):
        super().reset()
        self.mpc = mpc_control()

        return self._get_obs(), copy.deepcopy(self.state_info)

    def step(self, action):
        current_action = np.array(action) # + self.last_action
        vel, yaw = current_action
        vel = np.clip(vel, 0, 20)

        ego_x, ego_y = self._get_ego_pos()
        _, _, ego_yaw = self._get_delta_yaw()
        ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
        velocity = self.ego.get_velocity()
        vel_norm = np.linalg.norm(np.array([velocity.x, velocity.y]))
        current_wpt, _ = self._get_waypoint_xyz()
        self.mpc.feedbackCallback(ego_x, 
            ego_y, ego_heading, 
            vel_norm, self.mpc.str, 
            current_wpt[0], current_wpt[1], 
            yaw, vel)

        act = carla.VehicleControl(
            throttle=float(self.mpc.thr),
            steer=float(self.mpc.str * -pi / 3),
            brake=float(self.mpc.brk))
        self.ego.apply_control(act)

        for _ in range(4):
            self.world.tick()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        self.last_action = current_action

        # calculate reward
        isDone = self._terminal()
        current_reward = self._get_reward(np.array(current_action))

        return (self._get_obs(), current_reward, isDone,
                copy.deepcopy(self.state_info))
