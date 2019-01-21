import subprocess

from os import path
from time import sleep
from enum import Enum
from typing import Union

import airsim

from as_settings import AS_Settings

class Environments(Enum):
    Blocks       = 0
    Maze         = 1
    Neighborhood = 2



class AS_Environment:

    def __init__(self, target_positions,  env:Environments=Environments.Maze , env_path:str='env', drone:bool=True,
                 settings_path:str=None,
                 stepped_simulation:bool=True, step_duration:float=0.5,
                 manual_mode:bool=False, joystick:Union[int,None]=0,
                 rendering:bool=True, lowres:bool=True,
                 crash_terminate=True, max_steps = 1000, min_reward=-1000):

        self.settings = AS_Settings(settings_path)

        environments = {
            Environments.Blocks      : {'subfolder': ''        , 'bin': ''},
            Environments.Maze        : {'subfolder': 'Car_Maze', 'bin': 'Car_Maze.exe'},
            Environments.Neighborhood: {'subfolder': 'AirSimNH', 'bin': 'AirSimNH.exe'},
        }

        self.env = environments[env]

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
        if self.drone:
            self.client = airsim.MultirotorClient()
            self.settings.set('SimMode', 'Multirotor')
            self.settings.clear('Vehicles', True)

            vehicle_settings_path = 'Vehicles/SimpleFlight/'
            vehicle_type = 'SimpleFlight'

        else:
            self.client = airsim.CarClient()
            self.settings.set('SimMode', 'Car')
            self.settings.clear('Vehicles', True)

            vehicle_settings_path = 'Vehicles/PhysXCar/'
            vehicle_type = 'PhysXCar'

        self.settings.set(vehicle_settings_path + 'VehicleType', vehicle_type)
        self.settings.set(vehicle_settings_path + 'DefaultVehicleState', 'Armed')
        self.settings.set(vehicle_settings_path + 'AutoCreate', True)
        self.settings.set(vehicle_settings_path + 'EnableCollisionPassthrogh', False)
        self.settings.set(vehicle_settings_path + 'EnableCollision', True)
        self.settings.set(vehicle_settings_path + 'AllowAPIAlways', self.manual_mode)


        self.process_args['windowed'] =''


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

        self.acc_reward = 0
        self.min_reward = min_reward

        self.settings.dump()


    def reset(self, hard_reset=False, starting_position=(0, 0, 0)):
        if hard_reset:
            self.kill()

        if not self._env_running() or hard_reset:
            self._start()
            sleep(20)
            self.client.confirmConnection()
        else:
            # Check if env was already running
            self.client.ping()

        self.client.reset()
        self.client.enableApiControl(self.manual_mode)
        self.client.armDisarm(True)
        self.client.simPause(self.stepped_simulation)

        self.client.moveToPositionAsync(starting_position[0], starting_position[1], starting_position[2], 2).join()


    def kill(self):
        if self._env_running():
            self.process.kill()

    def _env_running(self):
        if self.process is not None:
            if self.process.poll() is not None:
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

    def step(self, action):

        self.steps += 1

        if self.stepped_simulation:
            self.client.simContinueForTime(self.step_duration)

        if self.drone:
            self.client.moveByVelocityZAsync()
        else:
            self.client.moveByVelocity()

        state = self._get_state()
        step_reward = self._reward_function(state)
        terminal = self._terminal_state(state)

        self.acc_reward += reward

        return state, step_reward, terminal


    def _get_state(self):
        camera_view = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        camera_view = camera_view[0]
        pose = self.client.simGetPose()

        collision_info = self.client.getCollisionInfo()

        state = {'scene': camera_view,
                 'pose' : pose,
                 'coll' : collision_info}

        return state


    def _reward_function(self, state):
        reward = 0

        current_target = self.target_positions.pop()

        planar_distance = sqrt()

        return reward

    def _terminal_state(self, state):

        terminal = False

        if state['coll']['force']:
            terminal = True

        if self.steps > self.max_steps:
            terminal = True

        if self.acc_reward < self.min_reward:
            terminal = True



        return terminal






# Class Tests
if __name__ == '__main__':

    env_path = path.join('..', '..', '..', 'env')

    env = AS_Environment(env=Environments.Maze, env_path=env_path)

    max_episodes = 3
    max_steps = 100

    for episode in range(max_episodes):
        env.reset()

        for step in range(max_steps):
            action = [10]
            state, reward, terminal = env.step(action)