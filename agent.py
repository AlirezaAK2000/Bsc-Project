# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
import time
import cv2
import math
import carla
from math import pi
import json

SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
EPISODE_LENGTH = 30







class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)  # seconds

        self.world = self.client.get_world()  # world connection
        self.blueprint_library = self.world.get_blueprint_library()

        self.model = self.blueprint_library.filter("bmw")[0]  # fetch the first model of bmw

        with open("carla/path.json" , 'r') as f:
            path = json.load(f)
            self.mapp = np.array([p['x'] for p in path]), np.array([p['y'] for p in path])
        
    

        
    def angular_error(self, yaw, x, y, n_x, n_y):
        theta_star = math.atan2(n_y - y, n_x - x)
        delta_theta = theta_star - yaw
        delta_theta = self.normalize_angle(delta_theta)
        return delta_theta
        
        

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        
        self.point_i = 0

        # spawn vehicle
        # self.vehicle_spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle_spawn_point = self.world.get_map().get_spawn_points()[28]  # training on a specific spawn point
        self.vehicle = self.world.spawn_actor(self.model, self.vehicle_spawn_point)
        self.vehicle.set_autopilot(False)  # making sure its not in autopilot!
        self.actor_list.append(self.vehicle)

        # camera sensory data
        self.camera = self.blueprint_library.find("sensor.camera.rgb")
        self.camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.camera.set_attribute("fov", "110")

        # spawn camera
        self.camera_spawn_point = carla.Transform(carla.Location(x=2, z=1))  # TODO: fine-tune these values!
        self.camera_sensor = self.world.spawn_actor(self.camera, self.camera_spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: self.process_camera_sensory_data(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(3)

        # collision sensor
        collison_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collison_sensor, self.camera_spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.process_collision_sensory_data(event))

        while self.front_camera is None:
            time.sleep(0.01)  # wait a little

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        t = self.vehicle.get_transform()
        x, y = t.location.x, t.location.y
        yaw = t.rotation.yaw
                
        return (x, y, yaw)


    def normalize_angle(self, angle):
        res = angle
        while res > pi:
            res -= 2.0 * pi
        while res < -pi:
            res += 2.0 * pi
        return res

    def process_camera_sensory_data(self, data):
        data_arr = np.array(data.raw_data, dtype=np.float64)
        data_pic = data_arr.reshape((self.im_height, self.im_width, 4))[:, :, :3]  # we only want rgb!
        if self.SHOW_CAM:
            cv2.imshow("", data_pic)
            cv2.waitKey(1)
        data_pic /= 255  # normalizing for the neural network
        self.front_camera = data_pic

    def process_collision_sensory_data(self, event):
        self.collision_hist.append(event)  # add the accident to the list

    def step(self, action, rrt_mode=False):
        t = self.vehicle.get_transform()
        x, y = t.location.x, t.location.y
        yaw = t.rotation.yaw

        # left action
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-1 * self.STEER_AMT))

        # straight
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))

        # right action 
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=1 * self.STEER_AMT))

        # nothing
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
        

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))


        # if we had a crash
        if len(self.collision_hist) != 0:
            done = True
            reward = -100

        # the car is moving too slow (discouraging running in circles even more!)
        elif kmh < 10:
            done = False
            reward = -5

        else:
            if i == len(self.mapp[0]) - 1:
                done = True
                reward = 100
            else:
                n_x, n_y = self.mapp[0][i], self.mapp[1][i]
                lateral_error = math.sqrt((x - n_x)**2 + (n_y - y)**2)
                long_error = self.angular_error(yaw ,x, y, n_x, n_y)
                #  terminating the episode (no reward)
                if self.episode_start + EPISODE_LENGTH < time.time():
                    done = True
                    
                if lateral_error
            
            

        return self.front_camera, reward, done, None