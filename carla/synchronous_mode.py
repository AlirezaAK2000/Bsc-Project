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
        vehicle.set_simulate_physics(True)


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
                
                image_semseg.save_to_disk("../data/msn/semantic/%06d.png" % current_frame)
                image_depth.save_to_disk("../data/msn/depth/%06d.png" % current_frame)
                image_rgb.save_to_disk("../data/msn/rgb/%06d.png" % current_frame)
                
                
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
