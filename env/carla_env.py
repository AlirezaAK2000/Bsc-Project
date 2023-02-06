
import collections
import queue
import time
import math
from math import pi
from env.utils import *
from env.config import *
import matplotlib.pyplot as plt
import numpy as np
import carla
import random
from functools import reduce

STRAIGHT = 0
NOTHING = 1
BRAKE = 2 

class CarlaEnv(object):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def __init__(self, conf, continuous_action = False, **kwargs):
        
        self.SHOW_CAM = conf['SHOW_PREVIEW']
        self.im_width = conf['IM_WIDTH']
        self.im_height = conf['IM_HEIGHT']
        self.frame_num = conf['FRAME_NUM']
        self.episode_length = conf['EPISODE_LENGTH']
        self.num_pedestrians = conf['NUM_PEDESTRIANS']
        self.v_min = conf['V_MIN']
        self.v_max = conf['V_MAX']
        self.ped_th = conf['PEDESTRIAN_CAMERA_RATIO_TH']
        self.ped_coef = conf['PED_COEF']
        self._town_name = conf['town']
        self.vehicle_name = conf['VEHICLE_NAME']
        
        
        
        self._client = carla.Client('localhost', conf['port'])
        self._client.set_timeout(30.0)
        
        

        set_sync_mode(self._client, False)

        self._world = self._client.load_world(self._town_name)
        self._map = self._world.get_map()
        self.continuous_action = continuous_action

        self._blueprints = self._world.get_blueprint_library()

        self._tick = 0
        self._player = None
        self._collision_reward = -100
        
        ax = plt.subplot(111)
        self.prog_shower = ax.imshow(np.zeros((self.im_height , self.im_width, 1)),cmap='gray', vmin=0, vmax=len(main_classes))
        plt.ion()

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)
        self._cameras = dict()
        self._routes = [0,1,6,8]
        
        self.throttle_min = 0
        self.throttle_max = 1
        self.brake_min = 0
        self.brake_max = 1
        self.n_action = 3
        
        

    def _set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)
        

    def reset(self, weather='random', seed=0):
        is_ready = False

        
        while not is_ready:
            np.random.seed(seed)

            self._clean_up()
            print(f'there are {len(self._world.get_map().get_spawn_points())} points')
            # self._spawn_player(random.choice(self._world.get_map().get_spawn_points()))
            self._spawn_player(self._world.get_map().get_spawn_points()[random.choice(self._routes)])
            self._setup_sensors()

            # self._set_weather(weather)
            self._pedestrian_pool = PedestrianPool(self._client, self.num_pedestrians)
            # self._vehicle_pool = VehiclePool(self._client, n_pedestrians)
            self._client.start_recorder("log.log")
                
                
            is_ready = self.ready()
        
        state = np.zeros((self.im_height, self.im_width, self.frame_num))
        
        for i in range(self.frame_num):
            
            self._world.tick()
            self._tick += 1
            self._pedestrian_pool.tick()
            
            img = self._cameras['sem_img'].get()
            state[:,:,i] = img.squeeze()
        
        return state.transpose((2, 0, 1))[None,:,:,:]
        

    def _spawn_player(self, start_pose):
        vehicle_bp = self._blueprints.filter(self.vehicle_name)[0]
        vehicle_bp.set_attribute('role_name', 'hero')
        self._player = self._world.spawn_actor(vehicle_bp, start_pose)
        self._actor_dict['player'].append(self._player)

    def ready(self, ticks=10):
        
        if not self.continuous_action:
            for _ in range(ticks):
                self.step(NOTHING)
        else:
            for _ in range(ticks):
                self.step([0,0])

        for x in self._actor_dict['camera']:
            x.get()

        self._time_start = time.time()
        self._tick = 0

        return True
    
    def step(self, action, increment_tick = True):

        if not self.continuous_action:

            if action == STRAIGHT:
                self._player.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
            elif action == NOTHING:
                self._player.apply_control(carla.VehicleControl(throttle=0, steer=0))
            elif action == BRAKE:
                self._player.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
            else:
                msg = f"There is no such thing {action}!!!"
                raise ValueError(msg)
        
        else:
            
            if action[0] > self.throttle_max or action[0] < self.throttle_min: 
                raise ValueError("Throttle must be between 0 and 1")
            if action[1] > self.brake_max or action[1] < self.brake_min: 
                raise ValueError("Brake must be between 0 and 1")
            
            self._player.apply_control(carla.VehicleControl(throttle=action[0], brake=action[1]))

        
        state = np.zeros((self.im_height, self.im_width, self.frame_num))
        done = False
        reward = 0
        
        info = dict()
        info['linear_speeds'] = []
        info['locs'] = []
        info['ang_speeds'] = []
        info['col_actor'] = None
        info['lane_invasion'] = False
        
        for i in range(self.frame_num):
            
            self._world.tick()
            if increment_tick:
                self._tick += 1
            self._pedestrian_pool.tick()
            
            img = self._cameras['sem_img'].get()
            state[:,:,i] = img.squeeze()
            
            if self.SHOW_CAM:
                self.prog_shower.set_data(img)
                plt.pause(0.0000001)
            
            transform = self._player.get_transform()
            v = self._player.get_velocity()
            kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
            
            x = transform.location.x
            y = transform.location.y
            theta = transform.rotation.yaw
            
            info['linear_speeds'].append(kmh)
            info['locs'].append((x, y))
            info['ang_speeds'].append(theta)
            info['nearest_ped'] = self._pedestrian_pool.check_dist(x, y)
            
            if len(self._collision_hist) > 0:
                other_actor = self._collision_hist[0].other_actor
                info['col_actor'] = other_actor
            
            if len(self._lane_invasion_hist) > 0 or (not np.any(img == main_classes['Road'])):
                info['lane_invasion'] = True
                    
        info['tick'] = self._tick
        reward , done= self._check_reward_and_termination(state, info)
        return state.transpose((2, 0, 1))[None,:,:,:] , reward, done, info
        

    def _speed_reward(self, v):
        
        reward = 0
        if v <= self.v_max:
            reward = np.exp((v - self.v_min) / 100) - 1
        else:
            reward = -np.exp(v / 100)
        return reward
    
    
    def _pixel_reward(self, frame):
        
        ped_pix = np.count_nonzero(frame == main_classes['Pedestrian'])
        ratio = ped_pix / reduce(lambda x,y: x*y, frame.shape)
        # print(f'The ratio of peds is {ratio} ')
        if ratio >= self.ped_th:
            return ratio * self.ped_coef
        return 0
        

    def _check_reward_and_termination(self, frames, info):
        
        done, reward = False, 0
        info['col_with_ped'] = False
        
        if info['nearest_ped'] < 2.3 or ( info['col_actor'] is not None and isinstance(info['col_actor'], carla.Walker)):
            reward = -100
            done = True
            info['col_with_ped'] = True
            return reward, done
        
        if (info['col_actor'] is not None) or info['lane_invasion']:
            done = True
            return reward, done
                   
        for i in range(frames.shape[-1]):
            reward += self._pixel_reward(frames[:,:,i]) / frames.shape[-1]
        
        # speed reward
        if reward == 0:
            reward += sum(map(self._speed_reward, info['linear_speeds']))/len(info['linear_speeds'])
        else:
            reward += sum(map(lambda x: -np.exp(x/100) + 1, info['linear_speeds']))/len(info['linear_speeds'])
            
        if self._tick / self.frame_num >= self.episode_length:
            done = True
            
        return reward, done
    
    def _process_collision_sensory_data(self, event):
        self._collision_hist.append(event)  # add the accident to the list

    def _process_lane_invasion_sensory_date(self, event):
        self._lane_invasion_hist.append(event)
    
    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        self._cameras['sem_img'] = Camera(self._world, self._player, self.im_width, self.im_height, 90, 1.8, 0.0, 1, 0.0, 0.0, type='semantic_segmentation')
        
        # collision sensor
        collison_sensor = self._blueprints.find("sensor.other.collision")
        collision_sensor = self._world.spawn_actor(collison_sensor, carla.Transform(), attach_to=self._player)
        collision_sensor.listen(lambda event: self._process_collision_sensory_data(event))
        self._actor_dict['collision_sensor'] = collision_sensor
        
        # lane invasion sensor
        li_sensor = self._blueprints.find("sensor.other.lane_invasion")
        li_sensor = self._world.spawn_actor(li_sensor, carla.Transform(), attach_to=self._player)
        li_sensor.listen(lambda event: self._process_lane_invasion_sensory_date(event))
        self._actor_dict['lane_invasion'] = li_sensor
        

    def __enter__(self):
        set_sync_mode(self._client, True)

        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self._clean_up()

        set_sync_mode(self._client, False)

    def _clean_up(self):
        self._pedestrian_pool = None
        self._vehicle_pool = None
        self._cameras.clear()

        
        for actor_type in list(self._actor_dict.keys()):
            
            if 'collision' in actor_type:
                self._actor_dict[actor_type].destroy()
                continue
            elif 'lane_invasion' in actor_type:
                self._actor_dict[actor_type].destroy()
                continue
            
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()

        self._tick = 0
        self._time_start = time.time()

        self._player = None
        self._collision_hist = []
        self._lane_invasion_hist = []
