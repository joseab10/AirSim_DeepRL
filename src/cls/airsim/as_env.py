import subprocess

from sys import platform
from os import path
from time import sleep
from enum import Enum
from typing import Union

import numpy as np
import airsim

from as_settings import AS_Settings

class Environments(Enum):
    Blocks       = 0
    Maze         = 1
    Neighborhood = 2

# CONSTANTS

class DroneActIdx(Enum):
    Pitch    = 0
    Yaw      = 1
    Roll     = 2
    Throttle = 3

class DroneAct(Enum):
    PitchFW = 0
    PitchBW = 1
    RollLF  = 2
    RollRG  = 3
    YawLF   = 4
    YawRG   = 5
    ThroUP  = 6
    ThroDW  = 7

class CarActIdx(Enum):
    VX       = 0
    VY       = 1
    VZ       = 2

class CarAct(Enum):
    VxFW     = 0
    VxBW     = 1
    VyFW     = 2
    VyBW     = 3
    VzFW     = 4
    VzBW     = 5


X = 0
Y = 1
Z = 2


class AS_Environment:

    def __init__(self, target_positions,  env:Environments=Environments.Maze , env_path:str='env', drone:bool=True,
                 settings_path:str=None,
                 stepped_simulation:bool=True, step_duration:float=0.1,
                 manual_mode:bool=False, joystick:Union[int,None]=0,
                 rendering:bool=True, lowres:bool=True,
                 crash_terminate=True, max_steps = 1000, min_reward=-1000):

        self.settings = AS_Settings(settings_path)

        environments = {
            Environments.Blocks      : {'subfolder': ''        , 'bin': ''},
            Environments.Maze        : {'subfolder': 'Car_Maze', 'bin': 'Car_Maze.exe'},
            Environments.Neighborhood: {'subfolder': 'AirSimNH', 'bin': 'AirSimNH.exe'},
        }

        #self.env = environments[env]

        self.env = ''
        for subpath in env_path.split(path.sep):
            self.env = path.join(self.env, subpath)
        for subpath in environments[env]['subfolder'].split(path.sep):
            self.env = path.join(self.env, subpath)
        self.env = path.join(self.env, environments[env]['bin'])

        # Process Object
        self.process = None
        self.process_args = {}

        self.drone = drone

        self.step_duration = step_duration

        self.manual_mode = not manual_mode
        self.stepped_simulation = stepped_simulation

        # AirSim API Client
        self.timeout = 15

        self.max_action_value = 250

        if not self.drone:
            self.client = airsim.CarClient(timeout_value=self.timeout)
            self.settings.set('SimMode', 'Car')
            self.settings.clear('Vehicles', True)

            self.vehicle_name = 'PhysXCar'
            vehicle_type = 'PhysXCar'

            self.prev_act = [0, 0, 0]

            # Action conversion settings
            v_factor = 5

            self.possible_actions = {
                CarAct.VxFW: {'index': CarActIdx.VX, 'sign':  1, 'factor': v_factor, 'str': 'Vx Fwd'},
                CarAct.VxBW: {'index': CarActIdx.VX, 'sign': -1, 'factor': v_factor, 'str': 'Vx Bwd'},
                CarAct.VyFW: {'index': CarActIdx.VY, 'sign':  1, 'factor': v_factor, 'str': 'Vy Fwd'},
                CarAct.VyBW: {'index': CarActIdx.VY, 'sign': -1, 'factor': v_factor, 'str': 'Vy Bwd'},
                CarAct.VzFW: {'index': CarActIdx.VZ, 'sign':  1, 'factor': v_factor, 'str': 'Vz Fwd'},
                CarAct.VzBW: {'index': CarActIdx.VZ, 'sign': -1, 'factor': v_factor, 'str': 'Vz Bwd'},
            }

        else:
            self.client = airsim.MultirotorClient(timeout_value=self.timeout)
            self.settings.set('SimMode', 'Multirotor')
            self.settings.clear('Vehicles', True)

            self.vehicle_name = 'SimpleFlight'
            vehicle_type = 'SimpleFlight'

            self.prev_act = [0, 0, 0, 0]

            # Action conversion settings
            pitch_roll_factor = 5
            yaw_factor = 10
            throttle_factor = 15

            self.possible_actions = {
                    DroneAct.PitchFW: {'index': DroneActIdx.Pitch   , 'sign':  1, 'factor': pitch_roll_factor, 'str': 'Pitch Fwd'},
                    DroneAct.PitchBW: {'index': DroneActIdx.Pitch   , 'sign': -1, 'factor': pitch_roll_factor, 'str': 'Pitch Bwd'},
                    DroneAct.RollRG : {'index': DroneActIdx.Roll    , 'sign':  1, 'factor': pitch_roll_factor, 'str': 'Roll Rght'},
                    DroneAct.RollLF : {'index': DroneActIdx.Roll    , 'sign': -1, 'factor': pitch_roll_factor, 'str': 'Roll Left'},
                    DroneAct.YawRG  : {'index': DroneActIdx.Yaw     , 'sign':  1, 'factor': yaw_factor       , 'str': 'Yaw Right'},
                    DroneAct.YawLF  : {'index': DroneActIdx.Yaw     , 'sign': -1, 'factor': yaw_factor       , 'str': 'Yaw  Left'},
                    DroneAct.ThroUP : {'index': DroneActIdx.Throttle, 'sign':  1, 'factor': throttle_factor  , 'str': 'Thrt   Up'},
                    DroneAct.ThroDW : {'index': DroneActIdx.Throttle, 'sign': -1, 'factor': throttle_factor  , 'str': 'Thrt  Dwn'},
            }

        vehicle_settings_path = 'Vehicles/' + self.vehicle_name + '/'
        self.settings.set(vehicle_settings_path + 'VehicleType', vehicle_type)
        self.settings.set(vehicle_settings_path + 'DefaultVehicleState', 'Armed')
        self.settings.set(vehicle_settings_path + 'AutoCreate', True)
        self.settings.set(vehicle_settings_path + 'EnableCollisionPassthrogh', False)
        self.settings.set(vehicle_settings_path + 'EnableCollision', True)
        self.settings.set(vehicle_settings_path + 'AllowAPIAlways', self.manual_mode)

        if not rendering:
            self.settings.set('ViewMode', 'NoDisplay')
            self.settings.set('SubWindows', [])
        else:
            self.settings.set('ViewMode', 'SpringArmChase')



        self.process_args['windowed'] = ''


        if lowres:
            self.process_args['ResX'] = 640
            self.process_args['ResY'] = 480

        if joystick is not None:
            self.settings.set(vehicle_settings_path + 'RC/RemoteControlID', joystick)
            self.settings.set(vehicle_settings_path + 'RC/AllowAPIWhenDisconnected', self.manual_mode)

        # Reward Terminal Conditions
        self.crash_terminate = crash_terminate
        self.target_positions = target_positions

        self.max_steps = max_steps
        self.steps = 0

        self.last_dist_to_target = 0

        self.acc_reward = 0
        self.min_reward = min_reward

        self.step_penalty = -10

        self.collission_threshold = 5



        self.settings.dump()


    def reset(self, hard_reset=False, starting_position=(0, 0, 0)):

        if hard_reset and self._env_running():
            self.close()

        env_found = False
        try:
            env_found = self.client.ping()
        except:
            pass

        if hard_reset or not(env_found or self._env_running()):
            self._start()
            sleep(5)
            if not self.drone:
                self.client = airsim.CarClient(timeout_value=self.timeout)
            else:
                self.client = airsim.MultirotorClient(timeout_value=self.timeout)
            self.client.confirmConnection()

        self.client.reset()
        self.client.enableApiControl(self.manual_mode)
        self.client.armDisarm(True)

        # Move to Initial Position
        tmp_pose = self.client.simGetVehiclePose()
        tmp_pose.position.x_val = starting_position[X]
        tmp_pose.position.y_val = starting_position[Y]
        tmp_pose.position.z_val = starting_position[Z]
        tmp_pose.orientation.x_val = starting_position[X]
        tmp_pose.orientation.y_val = starting_position[Y]
        tmp_pose.orientation.z_val = starting_position[Z]
        self.client.simSetVehiclePose(tmp_pose, True)

        if self.stepped_simulation:
            self.client.simPause(True)



        last_pos = starting_position
        dist_accum = 0
        max_points = 10000
        first_target = True
        for target in self.target_positions:
            tmp_dis = np.sqrt((last_pos[X] - target[X]) ** 2 +
                                 (last_pos[Y] - target[Y]) ** 2 +
                                 (last_pos[Z] - target[Z]) ** 2 )
            dist_accum += tmp_dis

            if first_target:
                self.last_dist_to_target = tmp_dis
                first_target = False

            last_pos = target
        self.points_per_meter = max_points / dist_accum
        self.min_reward = -3 * max_points



    def close(self):

        self.client.enableApiControl(False)

        if self._env_running():
            if platform == 'win32':
                pid = self.process.pid
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)])
            else:
                self.process.kill()


    def _env_running(self):
        if self.process is not None:
            if self.process.poll() is None:
                return True

        return False

    def _start(self):
        args = [self.env]

        for key, val in self.process_args.items():
            tmp_arg = '-' + key

            if val is not None and val != '':
                tmp_arg += '=' + str(val)

            args.append(tmp_arg)

        self.process = subprocess.Popen(args)

    def step(self, action_id):

        self.steps += 1

        action = self.id2action(action_id)

        if self.stepped_simulation:
            self.client.simPause(False)

        if self.drone:
            self.client.moveByAngleThrottleAsync(action[DroneActIdx.Pitch.value], action[DroneActIdx.Roll.value],
                                                 action[DroneActIdx.Throttle.value], action[DroneActIdx.Yaw.value],
                                                 self.step_duration).join()

        else:
            self.client.moveByVelocity(action[CarActIdx.VX.value], action[CarActIdx.VY.value], action[CarActIdx.VZ.value],
                                       self.step_duration).join()

        if self.stepped_simulation:
            self.client.simContinueForTime(self.step_duration)

        state = self._get_state()
        step_reward = self._reward_function(state)
        terminal = self._terminal_state(state)

        self.acc_reward += step_reward

        return state, step_reward, terminal


    def _get_state(self):
        camera_view = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        camera_view = camera_view[0]
        pose = self.client.simGetVehiclePose()

        collision_info = self.client.simGetCollisionInfo()

        state = {'scene': camera_view,
                 'pose' : pose,
                 'coll' : collision_info}

        return state


    def _reward_function(self, state):
        reward = 0

        current_target = self.target_positions[0]
        current_position = state['pose'].position
        current_position = (current_position.x_val, current_position.y_val, current_position.z_val)

        distance = np.sqrt((current_target[X] - current_position[X]) ** 2 +
                           (current_target[Y] - current_position[Y]) ** 2 +
                           (current_target[Z] - current_position[Z]) ** 2)

        advanced_distance = self.last_dist_to_target - distance
        reward += advanced_distance * self.points_per_meter

        self.last_dist_to_target = distance

        threshold = 0.1
        if distance < threshold:
            self.target_positions = self.target_positions[1:]

        collission = state['coll']
        if collission.has_collided:
            if collission.penetration_depth > self.collission_threshold:
                reward -= 200 * collission.penetration_depth

        reward += self.step_penalty

        return reward

    def _terminal_state(self, state):

        collission = state['coll']
        if collission.has_collided:
            if collission.penetration_depth > self.collission_threshold:
                return True

        if self.steps > self.max_steps:
            return True

        if self.acc_reward < self.min_reward:
            return True

        return False

    def _parse_actid(self, action_id:int):
        if self.drone:
            return DroneAct(action_id)
        else:
            return CarAct(action_id)


    def actid2str(self, action_id):
        action_id = self._parse_actid(action_id)

        return self.possible_actions[action_id]['str']

    def id2action(self, action_id):

        action_id = self._parse_actid(action_id)

        action = np.zeros(4)
        act = self.possible_actions[action_id]

        tmp_act = self.prev_act[act['index'].value] + act['sign'] * act['factor']
        if tmp_act < self.max_action_value:
            action[act['index'].value] = tmp_act
        else:
            action[act['index'].value] = self.max_action_value

        self.prev_act = action

        return action







# Class Tests
if __name__ == '__main__':


    env_path = path.join('..', '..', '..', 'env')

    target_positions = [(0, 0, 0), (1, 1, -1)]

    env = AS_Environment(target_positions, env=Environments.Maze, env_path=env_path,
                         stepped_simulation=True, step_duration=1)

    max_episodes = 3

    start_pos = [(0.5, 0.5, -2), (5, 5, -2), (3, 3, -2)]

    # Possible Actions
    a = [0, 1, 2, 3, 4, 5, 6, 7]
    # Action Probabilities
    p = [9, 9, 9, 9, 9, 9, 50, 0.1]
    p /= np.sum(p)

    for episode in range(max_episodes):
        env.reset(starting_position=start_pos[episode])

        terminal = False
        step = 1
        while not terminal:


            action_id = np.random.choice(a,p=p)
            action = env.actid2str(action_id)
            state, reward, terminal = env.step(action_id)
            print('Step: ', step, ' Distance: ', env.last_dist_to_target, ' Action: ', action_id, ' ',
                  action, ' Reward: ', reward, ' Total Reward: ', env.acc_reward )
            step +=1

    env.close()