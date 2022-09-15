#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
from turtle import width
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


classes = {
    'unlabled':np.uint8([[[0, 0, 0]]]), 
    'Building':np.uint8([[[70, 70, 70]]]), 
    'Fence':np.uint8([[[100, 40, 40]]]), 
    'Other':np.uint8([[[55, 90, 80]]]), 
    'Pedestrian':np.uint8([[[220, 20, 60]]]), 
    'Pole':np.uint8([[[153, 153, 153]]]), 	
    'RoadLine':np.uint8([[[157, 234, 50]]]), 
    'Road':np.uint8([[[128, 64, 128]]]), 
    'SideWalk':np.uint8([[[244, 35, 232]]]), 
    'Vegetation':np.uint8([[[107, 142, 35]]]), 
    'Vehicles':np.uint8([[[0, 0, 142]]]), 
    'Wall':np.uint8([[[102, 102, 156]]]), 
    'TrafficSign':np.uint8([[[220, 220, 0]]]), 
    'Sky':np.uint8([[[70, 130, 180]]]), 
    'Ground':np.uint8([[[81, 0, 81]]]), 
    'Bridge':np.uint8([[[150, 100, 100]]]), 
    'RailTrack':np.uint8([[[230, 150, 140]]]), 
    'GuardRail':np.uint8([[[180, 165, 180]]]), 
    'TrafficLight':np.uint8([[[250, 170, 30]]]), 
    'Static':np.uint8([[[110, 190, 160]]]), 
    'Dynamic':np.uint8([[[170, 120, 50]]]), 
    'Water':np.uint8([[[45, 60, 150]]]), 
    'Terrain':np.uint8([[[145, 170, 100]]]), 
}

anchor_classes = [
'Fence',
'Pole',
'Vegetation',
'TrafficSign',
'GuardRail',
'TrafficLight',
'Static',
'Terrain',
'Sky',
'Dynamic',
'Building',
'Wall',
'Water']

target_class = 'Other'

def change_class(img, anchor_class, target_class):
    mask = np.abs(img - anchor_class) <= 1
    mask = np.tile(np.prod(mask , axis=2)[:,:,None] , (1,1,3) )
    target_image = mask * target_class + (1 - mask) * img
    return target_image

def save_image(img, path):
    target_image = img
    target_image = target_image[:, :, :3]
    target_image = target_image[:, :, ::-1]

    for anchor_class in anchor_classes:
        target_image = change_class(target_image, classes[anchor_class], classes[target_class])
    target_image = Image.fromarray(target_image.astype(np.uint8))
    target_image.save(path)


def main():
    actor_list = []
    pygame.init()
    
    width, height = 256, 256

    display = pygame.display.set_mode(
        (width, height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    depth_display = pygame.display.set_mode(
        (width, height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.tesla.model3')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)


        # RGB camera 

        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(width))
        cam_bp.set_attribute("image_size_y",str(height))
        cam_bp.set_attribute("fov",str(105))

        camera_rgb = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(2.15,0,1), carla.Rotation(0,0,0)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        # semantic segmentation camera 
        
        sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute("image_size_x",str(width))
        sem_bp.set_attribute("image_size_y",str(height))
        sem_bp.set_attribute("fov",str(105))

        camera_semseg = world.spawn_actor(
            sem_bp,
            carla.Transform(carla.Location(2.15,0,1), carla.Rotation(0,0,0)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)
        
        
        
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute("image_size_x",str(width))
        depth_bp.set_attribute("image_size_y",str(height))
        depth_bp.set_attribute("fov",str(105))
        
        depth_cam = world.spawn_actor(
            depth_bp,
            carla.Transform(carla.Location(2.15,0,1), carla.Rotation(0,0,0)),
            attach_to=vehicle)
        actor_list.append(depth_cam)
        
        current_frame = 1

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, depth_cam, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg, image_depth = sync_mode.tick(timeout=2.0)
                

                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)

                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                image_depth.convert(carla.ColorConverter.LogarithmicDepth)
                
                save_image(np.frombuffer(image_semseg.raw_data, dtype=np.dtype("uint8")).reshape(256,256,4)
                           ,"../data/semantic256x256_test/data/%06d.png" % current_frame)
                # image_depth.save_to_disk("../data/msn/depth/%06d.png" % current_frame)
                # image_rgb.save_to_disk("../data/msn/rgb/%06d.png" % current_frame)
                
                
                assert image_semseg.frame == image_rgb.frame == image_depth.frame
                
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                display.blit(
                    font.render('% 5d Frame Number : ' % current_frame, True, (255, 255, 255)),
                    (8, 46))
                
                current_frame += 1
                pygame.display.flip()
                

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
