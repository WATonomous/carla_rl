#!/usr/bin/env python

# This file is modified by Dongjie yu (yudongjie.moon@foxmail.com)
# from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# author: Jianyu Chen (jianyuchen@berkeley.edu)

from __future__ import division
import copy
import numpy as np
import random
import time
from collections import deque
from carla import ColorConverter as cc
import pygame

import gym
from gym import spaces
from gym.utils import seeding
import carla
import cv2
from math import pi
from parl.utils import CSVLogger

from .misc import _vec_decompose, delta_angle_between
from .carla_logger import *
from .mpc import mpc_control

from .carla_env_orig import CarlaEnv


class CarlaEnvMPC(CarlaEnv):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        super(CarlaEnvMPC, self).__init__(params)
        # Main changes in __init__
        self.mpc = mpc_control()
        self.enable_ped = False

        #-------------------------------------
        self.host = params['host']
        self.logger = setup_carla_logger(
            "logs", experiment_name=params['exp_name'])
        self.logger.error("Env running in port {}".format(params['port']))
        self.csv_logger = CSVLogger(f"{params['exp_name']}.csv")
        # parameters
        self.dt = params['dt']
        self.port = params['port']
        self.task_mode = params['task_mode']
        self.code_mode = params['code_mode']
        self.max_time_episode = params['max_time_episode']

        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.enable_target = params['enable_target']

        # action and observation space
        self.action_space = spaces.Box(
            np.array([-2.0, -2.0]), np.array([2.0, 2.0]), dtype=np.float32)
        self.state_space = spaces.Box(
            low=-50.0, high=50.0, shape=(15, ), dtype=np.float32)

        # Connect to carla server and get world object
        # self.logger.info(('connecting to Carla server...')
        self._make_carla_client(self.host, self.port)

        # Load routes
        self.route_deterministic_id = 0

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(
            params['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find(
            'sensor.other.collision')

        self.CAM_RES = 1024
        # Add camera sensor
        self.camera_img = np.zeros((self.CAM_RES, self.CAM_RES, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=30), carla.Rotation(pitch=-90))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.CAM_RES))
        self.camera_bp.set_attribute('image_size_y', str(self.CAM_RES))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # A dict used for storing state data
        self.state_info = {}

        # A list stores the ids for each episode
        self.actors = []

        # Future distances to get heading
        self.distances = [1., 5., 10.]
        self.target_vehicle = None

    def reset(self):
        self.collision_sensor = None
        self.lane_sensor = None

        # Delete sensors, vehicles and walkers
        while self.actors:
            (self.actors.pop()).destroy()

        self._load_world()
        self.mpc = mpc_control()

        # Spawn the ego vehicle at a random position between start and dest
        # Start and Destination
        if self.task_mode == 'Straight':
            self.route_id = 0
        elif self.task_mode == 'Curve':
            self.route_id = 1  #np.random.randint(2, 4)
        elif self.task_mode == 'Long' or self.task_mode == 'Lane' or self.task_mode == 'Lane_test' or self.task_mode == 'Left_turn':
            if self.code_mode == 'train':
                self.route_id = np.random.randint(0, 4)
            elif self.code_mode == 'test':
                self.route_id = self.route_deterministic_id
                self.route_deterministic_id = (
                    self.route_deterministic_id + 1) % 4
        elif self.task_mode == 'U_curve':
            self.route_id = 0
        self.start = self.left_turn_wpts[0].transform
        self.dest = self.left_turn_wpts[-1].transform
        

        if self.enable_target:
            self.target_vehicle = self._try_spawn_random_vehicle()
            self.actors.append(self.target_vehicle)
        if self.enable_ped:
            self.ped = self._try_spawn_random_ped()
            self.actors.append(self.ped)
        ego_spawned = False
        start = self.start
        init_fwd_speed = 0
        while not ego_spawned:
            # Code_mode == train, spwan randomly between start and destination
            # if self.code_mode == 'train':
                # start, init_fwd_speed = self._get_random_position_between()
                # init_fwd_speed = 5 * np.random.random()
                # print(f"init_fwd_speed: {init_fwd_speed}")
            try:
                self._try_spawn_ego_vehicle_at(start)
                ego_spawned = True
            except:
                ego_spawned = False
                # print("ego failed to spawn, re-trying")
        yaw = (start.rotation.yaw) * np.pi / 180.0
        init_speed = carla.Vector3D(
                    x=init_fwd_speed * np.cos(yaw),
                    y=init_fwd_speed * np.sin(yaw))

        # Add collision sensor
        self.collision_sensor = self.world.try_spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.collision_sensor)
        self.collision_sensor.listen(
            lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 +
                                impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)


        def get_camera_img(data):
            self.og_camera_img = data
        self.collision_hist = []
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.actors.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: get_camera_img(data))


        # Update timesteps
        self.time_step = 1
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        if self.code_mode == "train":
            self.settings.no_rendering_mode = True
        else:
            self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)

        # Set the initial speed to desired speed
        self.ego.set_velocity(init_speed)
        physics = self.ego.get_physics_control()
        physics.gear_switch_time *= 0.0
        physics.use_gear_autobox = False
        self.ego.apply_physics_control(physics)
        for _ in range(2):
            self.world.tick()

        # Get waypoint infomation
        ego_x, ego_y = self._get_ego_pos()
        self.ego_x, self.ego_y = ego_x, ego_y
        self.current_wpt, progress = self._get_waypoint_xyz()

        delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
        road_heading = np.array([
            np.cos(wpt_yaw / 180 * np.pi),
            np.sin(wpt_yaw / 180 * np.pi)
        ])
        ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
        self.ego_heading = ego_heading
        ego_heading_vec = np.array(
            [np.cos(ego_heading),
                np.sin(ego_heading)])

        future_angles = self._get_future_wpt_angle(
            distances=self.distances)

        # Update State Info (Necessary?)
        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        dyaw_dt = self.ego.get_angular_velocity().z
        v_t_absolute = np.array([velocity.x, velocity.y])
        a_t_absolute = np.array([accel.x, accel.y])
        self.vel_norm = np.linalg.norm(np.array([velocity.x, velocity.y]))

        # decompose v and a to tangential and normal in ego coordinates
        v_t = _vec_decompose(v_t_absolute, ego_heading_vec)
        a_t = _vec_decompose(a_t_absolute, ego_heading_vec)

        # Reset action of last time step
        # TODO:[another kind of action]
        self.last_action = np.array([0.0, 0.0])

        pos_err_vec = np.array((ego_x, ego_y)) - self.current_wpt[0:2]

        self.state_info['velocity_t'] = v_t
        self.state_info['acceleration_t'] = a_t

        # self.state_info['ego_heading'] = ego_heading
        self.state_info['delta_yaw_t'] = delta_yaw
        self.state_info['dyaw_dt_t'] = dyaw_dt

        self.state_info['lateral_dist_t'] = np.linalg.norm(pos_err_vec) * \
                                            np.sign(pos_err_vec[0] * road_heading[1] - \
                                                    pos_err_vec[1] * road_heading[0])
        self.state_info['action_t_1'] = self.last_action
        self.state_info['angles_t'] = future_angles
        self.state_info['progress'] = progress
        self.state_info['target_vehicle_dist_y'] = 50
        self.state_info['target_vehicle_dist_x'] = 50
        self.state_info['target_vehicle_vel'] = 0

        if self.target_vehicle is not None:
            t_loc = self.target_vehicle.get_location()
            e_loc = self.ego.get_location()
            self.state_info['target_vehicle_dist_y'] = t_loc.y - e_loc.y
            self.state_info['target_vehicle_dist_x'] = e_loc.x - t_loc.x
            self.state_info['target_vehicle_vel'] = -1*self.target_vehicle.get_velocity().y
        self.state_info['ped_dist_y'] = 50
        self.state_info['ped_dist_x'] = 50
        self.state_info['ped_vel'] = 0
        if self.ped is not None:
            t_loc = self.ped.get_location()
            e_loc = self.ego.get_location()
            self.state_info['ped_dist_y'] = t_loc.y - e_loc.y
            self.state_info['ped_dist_x'] = e_loc.x - t_loc.x
            self.state_info['ped_vel'] = -1*self.ped.get_velocity().y

        # End State variable initialized
        self.isCollided = False
        self.isTimeOut = False
        self.isSuccess = False
        self.isOutOfLane = False
        self.isSpecialSpeed = False

        return self._get_obs(), copy.deepcopy(self.state_info)

    def step(self, action):

        # Assign acc/steer/brake to action signal
        # Ver. 1 input is the value of control signal
        # throttle_or_brake, steer = action[0], action[1]
        # if throttle_or_brake >= 0:
        #     throttle = throttle_or_brake
        #     brake = 0
        # else:
        #     throttle = 0
        #     brake = -throttle_or_brake

        # Ver. 2 input is the delta value of control signal
        # TODO:[another kind of action] change the action space to [-2, 2]
        
        current_action = np.array(action) # + self.last_action
        self.logger.info("Received action: {} current action: {}".format(action, current_action)) 
        # self.csv_logger.log_dict({"received action vel": action[0], 
        # "received action yaw": action[1],
        # "current action vel": current_action[0],
        # "current action yaw": current_action[1],
        # "ego x": self.ego_x,
        # "ego_y": self.ego_y,
        # "ego_heading": self.ego_heading,
        # "ego vel_norm": self.vel_norm,
        # "ego steering": self.mpc.str,
        # "waypoint x": self.current_wpt[0],
        # "waypoint y": self.current_wpt[1],
        # })
        vel, yaw = current_action
        vel = np.clip(vel, 0, 20)
        # vel = 8
        # yaw = self.ego_heading
        self.logger.info("MPC Input {}".format({self.ego_x, 
            self.ego_y, self.ego_heading, 
            self.vel_norm, self.mpc.str, 
            self.current_wpt[0], self.current_wpt[1], 
            yaw, vel}))

        self.mpc.feedbackCallback(self.ego_x, 
            self.ego_y, self.ego_heading, 
            self.vel_norm, self.mpc.str, 
            self.current_wpt[0], self.current_wpt[1], 
            yaw, vel)

        # throttle_or_brake, steer = current_action
        # if throttle_or_brake >= 0:
        #     throttle = throttle_or_brake
        #     brake = 0
        # else:
        #     throttle = 0
        #     brake = -throttle_or_brake

        # Apply control
        self.logger.info("Vehicle Control Input throttle: {}, steer: {}, brake: {}".format(
            float(self.mpc.thr),
            float(self.mpc.str * -pi / 3),
            float(self.mpc.brk)
            ))

        self.csv_logger.log_dict({
            "mpc throttle": float(self.mpc.thr), 
            "mpc steer": float(self.mpc.str * -pi / 3),
            "mpc brake": float(self.mpc.brk)
        })

        act = carla.VehicleControl(
            throttle=float(self.mpc.thr),
            steer=float(self.mpc.str * -pi / 3),
            brake=float(self.mpc.brk))

        self.logger.info("Applying control: {}".format(act)) 
        self.ego.apply_control(act)

        for _ in range(4):
            self.world.tick()
        self.logger.info("World ticks done")
        # Get waypoint infomation
        self.ego_x, self.ego_y = self._get_ego_pos()
        self.current_wpt, progress = self._get_waypoint_xyz()

        delta_yaw, wpt_yaw, ego_yaw = self._get_delta_yaw()
        road_heading = np.array(
            [np.cos(wpt_yaw / 180 * np.pi),
                np.sin(wpt_yaw / 180 * np.pi)])
        self.ego_heading = np.float32(ego_yaw / 180.0 * np.pi)
        ego_heading_vec = np.array((np.cos(self.ego_heading),
                                    np.sin(self.ego_heading)))

        future_angles = self._get_future_wpt_angle(
            distances=self.distances)

        # Get dynamics info
        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        dyaw_dt = self.ego.get_angular_velocity().z
        v_t_absolute = np.array([velocity.x, velocity.y])
        a_t_absolute = np.array([accel.x, accel.y])
        self.vel_norm = np.linalg.norm(np.array([velocity.x, velocity.y]))

        # decompose v and a to tangential and normal in ego coordinates
        v_t = _vec_decompose(v_t_absolute, ego_heading_vec)
        a_t = _vec_decompose(a_t_absolute, ego_heading_vec)

        pos_err_vec = np.array((self.ego_x, self.ego_y)) -  self.current_wpt[0:2]

        self.state_info['velocity_t'] = v_t
        self.state_info['acceleration_t'] = a_t

        # self.state_info['ego_heading'] = ego_heading
        self.state_info['delta_yaw_t'] = delta_yaw
        self.state_info['dyaw_dt_t'] = dyaw_dt

        self.state_info['lateral_dist_t'] = np.linalg.norm(pos_err_vec) * \
                                            np.sign(pos_err_vec[0] * road_heading[1] - \
                                                    pos_err_vec[1] * road_heading[0])
        self.state_info['action_t_1'] = self.last_action
        self.state_info['angles_t'] = future_angles
        self.state_info['progress'] = progress
        self.state_info['target_vehicle_dist_y'] = 50
        self.state_info['target_vehicle_dist_x'] = 50
        self.state_info['target_vehicle_vel'] = 0
        if self.target_vehicle is not None:
            t_loc = self.target_vehicle.get_location()
            e_loc = self.ego.get_location()
            self.state_info['target_vehicle_dist_y'] = t_loc.y - e_loc.y
            self.state_info['target_vehicle_dist_x'] = e_loc.x - t_loc.x
            self.state_info['target_vehicle_vel'] = -1*self.target_vehicle.get_velocity().y

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        self.last_action = current_action

        # calculate reward
        self.logger.info("Running _terminal & _get_reward")
        isDone = self._terminal()
        current_reward = self._get_reward(np.array(current_action))
        self.logger.info("Obtained Reward: {}".format(current_reward))

        return (self._get_obs(), current_reward, isDone,
                copy.deepcopy(self.state_info))

    def _info2normalized_state_vector(self):
        '''
        params: dict of ego state(velocity_t, accelearation_t, dist, command, delta_yaw_t, dyaw_dt_t)
        type: np.array
        return: array of size[9,], torch.Tensor (v_x, v_y, a_x, a_y
                                                 delta_yaw, dyaw, d_lateral, action_last,
                                                 future_angles)
        '''
        velocity_t = self.state_info['velocity_t'] / (self.desired_speed * 1.5)
        accel_t = self.state_info['acceleration_t'] / 40
        delta_yaw_t = np.array(self.state_info['delta_yaw_t']).reshape(
            (1, )) / 180
        dyaw_dt_t = np.array(self.state_info['dyaw_dt_t']).reshape((1, )) / 30.0
        lateral_dist_t = self.state_info['lateral_dist_t'].reshape(
            (1, )) / 5      
        action_last = self.state_info['action_t_1'] / 3

        future_angles = self.state_info['angles_t'] / 90
        target_dist_y = np.array(self.state_info['target_vehicle_dist_y']).reshape((1, )) / 40
        target_dist_x = np.array(self.state_info['target_vehicle_dist_x']).reshape((1, )) / 30
        target_vel = np.array(self.state_info['target_vehicle_vel']).reshape((1, )) / 7

        if (self.enable_ped):
            # Add pedestrian states if pedestrian is enabled
            ped_dist_y = np.array(self.state_info['ped_dist_y']).reshape((1, )) / 40
            ped_dist_x = np.array(self.state_info['ped_dist_x']).reshape((1, )) / 30
            ped_vel = np.array(self.state_info['ped_vel']).reshape((1, )) / 7
            info_vec = np.concatenate([
                velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, lateral_dist_t,
                action_last, future_angles, target_dist_y, target_dist_x, target_vel, ped_dist_y, ped_dist_x, ped_vel
            ], axis=0)
        else:
            info_vec = np.concatenate([
                velocity_t, accel_t, delta_yaw_t, dyaw_dt_t, lateral_dist_t,
                action_last, future_angles, target_dist_y, target_dist_x, target_vel
            ], axis=0)
        info_vec = info_vec.squeeze()

        return info_vec