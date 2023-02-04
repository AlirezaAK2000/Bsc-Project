# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

# import random
# import cv2


# class CarEnv:
#     SHOW_CAM = SHOW_PREVIEW
#     STEER_AMT = 1.0
#     im_width = IM_WIDTH
#     im_height = IM_HEIGHT
#     front_camera = None
#     # num_pedestrians = NUM_PEDESTRIANS
#     frame_num = FRAME_NUM
#     episode_length = EPISODE_LENGTH
    
#     def __init__(self, fps=20):
#         self.client = carla.Client("localhost", 2000)
#         self.client.set_timeout(4.0)  # seconds
#         set_sync_mode(self.client, False)        

#         self.world = self.client.load_world('Town02')  # world connection
        
        
#         self.blueprint_library = self.world.get_blueprint_library()

#         self.model = self.blueprint_library.filter("bmw")[0]  # fetch the first model of bmw


        
#         self.actor_list = []

#         # with open("carla/path.json" , 'r') as f:
#         #     path = json.load(f)
#         #     self.mapp = np.array([p['x'] for p in path]), np.array([p['y'] for p in path])
        
    

        
#     # def angular_error(self, yaw, x, y, n_x, n_y):
#     #     theta_star = math.atan2(n_y - y, n_x - x)
#     #     delta_theta = theta_star - yaw
#     #     delta_theta = self.normalize_angle(delta_theta)
#     #     return delta_theta
        
        

#     def reset(self):
        
#         if len(self.actor_list) > 0:
            
#             for actor from actor_list:
#                 actor.destroy()
        
        

#         self.actor_list = []
        
#         self.step_num = 1

#         # spawn vehicle
#         self.vehicle_spawn_point = random.choice(self.world.get_map().get_spawn_points())
#         # self.vehicle_spawn_point = self.world.get_map().get_spawn_points()[28]  # training on a specific spawn point
#         self.vehicle = self.world.spawn_actor(self.model, self.vehicle_spawn_point)
#         self.vehicle.set_autopilot(False)  # making sure its not in autopilot!
#         self.actor_list.append(self.vehicle)

#         # camera sensory data
#         self.camera = self.blueprint_library.find("sensor.camera.semantic_segmentation")
#         self.camera.set_attribute("image_size_x", f"{IM_WIDTH}")
#         self.camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
#         self.camera.set_attribute("fov", "110")

#         # spawn camera
#         self.camera_spawn_point = carla.Transform(carla.Location(x=2, z=1))  # TODO: fine-tune these values!
#         self.camera_sensor = self.world.spawn_actor(self.camera, self.camera_spawn_point, attach_to=self.vehicle)
#         self.actor_list.append(self.camera_sensor)
#         # self.camera_sensor.listen(lambda data: self.process_camera_sensory_data(data))
#         self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

#         #spawn Pedestrians
#         # # self._pedestrians = PedestrianPool(self.client, self.num_pedestrians)

#         time.sleep(3)
        

#         # collision sensor
#         collison_sensor = self.blueprint_library.find("sensor.other.collision")
#         self.collision_sensor = self.world.spawn_actor(collison_sensor, self.camera_spawn_point, attach_to=self.vehicle)
#         self.actor_list.append(self.collision_sensor)
#         self.collision_sensor.listen(lambda event: self._process_collision_sensory_data(event))

#         # while self.front_camera is None:
#         #     time.sleep(0.01)  # wait a little

#         self.episode_start = time.time()
#         self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

#         state = self._get_sync_state()
#         return state


#     # def normalize_angle(self, angle):
#     #     res = angle
#     #     while res > pi:
#     #         res -= 2.0 * pi
#     #     while res < -pi:
#     #         res += 2.0 * pi
#     #     return res

            
        

#     # def _process_collision_sensory_data(self, event):
#     #     self.collision_hist.append(event)  # add the accident to the list
        
    
#     def _check_reward_and_termiantion(self, frames, speeds):
        
#         done = False
        
#         if self.step_num == self.episode_length:
#             done = True
        
    
#         return (0, done)
    
    
#     def _get_sync_state(self):
        
#         collision_events = []
#         frames = np.zeros((self.im_height, self.im_width, self.frame_num))
#         speeds = []
#         with CarlaSyncMode(self.world, self.camera_sensor, fps=30) as sync_mode:
#             for i in range(self.frame_num):   
#                 snap_shot, img = sync_mode.tick(timeout = 2.0)
#                 img = self._process_camera_sensory_data(img)
#                 frames[:,:,i] = img.squeeze()
                
#                 if self.SHOW_CAM:
#                     self.prog_shower.set_data(img)
#                     plt.pause(0.0000001)
#                 # collision_events.append(event)
            
#                 v = self.vehicle.get_velocity()
#                 kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
#                 speeds.append(kmh)
                
        
#         return (speeds, frames, [])

#     def step(self, action):

#         if self.step_num > self.episode_length:
#             raise ValueError("Episode is finished.")

#         done = False
#         # left action
#         if action == 0:
#             self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-1 * self.STEER_AMT))

#         # straight
#         elif action == 1:
#             self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))

#         # right action 
#         elif action == 2:
#             self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=1 * self.STEER_AMT))

#         # nothing
#         elif action == 3:
#             self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
         
#         # brake    
#         elif action == 4:
#             self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=0))
        

#         # v = self.vehicle.get_velocity()
#         # kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))


#         # if we had a crash
#         # if len(self.collision_hist) != 0:
#         #     done = True
#         #     reward = -100

#         # the car is moving too slow (discouraging running in circles even more!)
#         # elif kmh < 10:
#         #     done = False
#         #     reward = -5
#         # self._pedestrians.tick()
#         # else:
#         #     if i == len(self.mapp[0]) - 1:
#         #         done = True
#         #         reward = 100
#         #     else:
#         #         n_x, n_y = self.mapp[0][i], self.mapp[1][i]
#         #         lateral_error = math.sqrt((x - n_x)**2 + (n_y - y)**2)
#         #         long_error = self.angular_error(yaw ,x, y, n_x, n_y)
#         #         #  terminating the episode (no reward)
#         #         if self.episode_start + EPISODE_LENGTH < time.time():
#         #             done = True
                    
#         #         if lateral_error
        
        

#         state = self._get_sync_state()
        
#         speeds, frames, collision_events = state
        
#         self.step_num += 1
        
#         reward, done = self._check_reward_and_termiantion(frames, speeds)

#         return state, reward, done, None



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

class CarlaEnv(object):
    
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    frame_num = FRAME_NUM
    episode_length = EPISODE_LENGTH
    num_pedestrians = NUM_PEDESTRIANS
    v_min = V_MIN
    v_max = V_MAX
    ped_th = PEDESTRIAN_CAMERA_RATIO_TH
    ped_coef = PED_COEF
    
    
    def __init__(self, town='Town02', port=2000, **kwargs):
        self._client = carla.Client('localhost', port)
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

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
        

    def _set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def reset(self, weather='random', seed=0):
        is_ready = False

        self.collision_hist = []
        
        while not is_ready:
            np.random.seed(seed)

            self._clean_up()
            print(f'there are {len(self._world.get_map().get_spawn_points())} points')
            self._spawn_player(random.choice(self._world.get_map().get_spawn_points()))
            self._setup_sensors()

            self._set_weather(weather)
            self._pedestrian_pool = PedestrianPool(self._client, self.num_pedestrians)
            # self._vehicle_pool = VehiclePool(self._client, n_pedestrians)

            is_ready = self.ready()
        
        state = np.zeros((self.im_height, self.im_width, self.frame_num))
        
        for i in range(self.frame_num):
            
            self._world.tick()
            self._tick += 1
            self._pedestrian_pool.tick()
            
            img = self._cameras['sem_img'].get()
            state[:,:,i] = img.squeeze()
            
        
        return state
        

    def _spawn_player(self, start_pose):
        vehicle_bp = self._blueprints.filter(VEHICLE_NAME)[0]
        vehicle_bp.set_attribute('role_name', 'hero')

        self._player = self._world.spawn_actor(vehicle_bp, start_pose)

        self._actor_dict['player'].append(self._player)

    def ready(self, ticks=10):
        for _ in range(ticks):
            self.step(3)

        for x in self._actor_dict['camera']:
            x.get()

        self._time_start = time.time()
        self._tick = 0

        return True
    

    def step(self, action):

        # left action
        if action == 0:
            self._player.apply_control(carla.VehicleControl(throttle=0.3, steer=-1 * self.STEER_AMT))

        # straight
        elif action == 1:
            self._player.apply_control(carla.VehicleControl(throttle=0.5, steer=0))

        # right action 
        elif action == 2:
            self._player.apply_control(carla.VehicleControl(throttle=0.3, steer=1 * self.STEER_AMT))

        # nothing
        elif action == 3:
            self._player.apply_control(carla.VehicleControl(throttle=0, steer=0))
         
        # brake    
        elif action == 4:
            self._player.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=0))
        


        # Put here for speed (get() busy polls queue).
        
        
        state = np.zeros((self.im_height, self.im_width, self.frame_num))
        done = False
        reward = 0
        
        info = dict()
        info['linear_speeds'] = []
        info['locs'] = []
        info['ang_speeds'] = []
        
        for i in range(self.frame_num):
            

            self._world.tick()
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
            
            if len(self.collision_hist) > 0:
                other_actor = self.collision_hist[0].other_actor
                info['col_actor'] = other_actor
                # if isinstance(other_actor, carla.Walker) and not ped_col:
                #     ped_col = True
                    
        
        info['tick'] = self._tick
        
        reward , done= self._check_reward_and_termination(state, info)
        
        return state, reward, done, info
        

    def _speed_reward(self, v):
        
        reward = 0
        
        if v <= self.v_max:
            reward = np.exp((v - v_min) / 100) - 1
        else:
            reward = -np.exp(v / 100)
        
        
        return reward
    
    def _pixel_reward(self, frame):
        
        ped_pix = np.count_nonzero(frame == base_classes['Pedestrian'])
        
        ratio = ped_pix / reduce(lambda x,y: x*y, frame.shape)
        
        if ratio >= self.ped_th:
            return ratio * self.ped_coef
        
        return 0
        

    def _check_reward_and_termination(self, frames, info):
        
        done, reward = False, 0
        
        if info['nearest_ped'] < 2.3 or ( info['col_actor'] is not None and isinstance(info['col_actor'], carla.Walker)):
            reward = -100
            done = True
            return reward, done
        
        if info['col_actor'] is not None and isinstance(info['col_actor'], carla.Walker):
            done = True
            return reward, done
                   
        # speed reward
        for i in range(frames.shape[-1]):
            reward += self._pixel_reward(frame[:,:,i]) / frames.shape[-1]
        
        if reward == 0:
            reward += sum(map(self._speed_reward, info['linear_speeds']))/len(info['linear_speeds'])
        else:
            reward += sum(map(lambda x: -np.exp(x), info['linear_speeds']))/len(info['linear_speeds'])
            
        if self._tick / self.frame_num >= self.episode_length:
            done = True
            
        return reward, done
    
    def _process_collision_sensory_data(self, event):
        self.collision_hist.append(event)  # add the accident to the list

    
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
                break
            self._client.apply_batch([carla.command.DestroyActor(x) for x in self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()

        self._tick = 0
        self._time_start = time.time()

        self._player = None
