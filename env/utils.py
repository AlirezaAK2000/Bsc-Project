import numpy as np
import carla
try:
    import queue
except ImportError:
    import Queue as queue
    
from PIL import Image, ImageDraw
from env.config import *
import math

def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0
    settings.no_rendering_mode = False

    world.apply_settings(settings)


class Camera(object):
    def __init__(self, world, player, w, h, fov, x, y, z, pitch, yaw, type='rgb'):
        bp = world.get_blueprint_library().find('sensor.camera.%s' % type)
        bp.set_attribute('image_size_x', str(w))
        bp.set_attribute('image_size_y', str(h))
        bp.set_attribute('fov', str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.type = type
        self.queue = queue.Queue()

        self.camera = world.spawn_actor(bp, transform, attach_to=player)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None

        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()
            
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = self._process_camera_sensory_data(array, image.height, image.width)
        
        # if self.type == 'semantic_segmentation':
        #     return array[:, :, 0]

        return array

    
    def _process_camera_sensory_data(self, data, height, width):
        
        data_arr = data.reshape((height , width,4))
        data_arr = data_arr[:, :, :3]
        data_pic = data_arr[:, :, ::-1]
        
        # data_pic = np.array(Image.fromarray(data_arr).convert('RGB'))

        d = np.zeros((height, width, 1))
        
        
        for k,v in main_classes.items():
            a = data_pic == base_classes[k]
            a = np.prod(a , axis=2)[:,:,None] 
            a = a == 1
            d[a] = v

        d = d.astype(np.uint8)
        return d

    def __del__(self):
        self.camera.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()



class VehiclePool(object):
    def __init__(self, client, n_vehicles):
        self.client = client
        self.world = client.get_world()

        veh_bp = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = np.random.choice(self.world.get_map().get_spawn_points(), n_vehicles)
        batch = list()

        for i, transform in enumerate(spawn_points):
            bp = np.random.choice(veh_bp)
            bp.set_attribute('role_name', 'autopilot')

            batch.append(
                    carla.command.SpawnActor(bp, transform).then(
                        carla.command.SetAutopilot(carla.command.FutureActor, True)))

        self.vehicles = list()
        errors = set()

        for msg in self.client.apply_batch_sync(batch):
            if msg.error:
                errors.add(msg.error)
            else:
                self.vehicles.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        print('%d / %d vehicles spawned.' % (len(self.vehicles), n_vehicles))

    def __del__(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])


class PedestrianPool(object):
    def __init__(self, client, n_pedestrians):
        self.client = client
        self.world = client.get_world()

        ped_bp = self.world.get_blueprint_library().find('walker.pedestrian.0002')
        # ped_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        con_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        
        # for actor in ped_bp:
            
        #     if actor.has_attribute('is_invincible'):
        #         actor.set_attribute('is_invincible', 'false')
                
        if ped_bp.has_attribute('is_invincible'):
            ped_bp.set_attribute('is_invincible', 'false')

        spawn_points = [self._get_spawn_point() for _ in range(n_pedestrians)]
        # batch = [carla.command.SpawnActor(np.random.choice(ped_bp), spawn) for spawn in spawn_points]
        batch = [carla.command.SpawnActor(ped_bp, spawn) for spawn in spawn_points]
        
        walkers = list()
        errors = set()

        for msg in client.apply_batch_sync(batch, True):
            if msg.error:
                errors.add(msg.error)
            else:
                walkers.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        batch = [carla.command.SpawnActor(con_bp, carla.Transform(), walker_id) for walker_id in walkers]
        controllers = list()
        errors = set()

        for msg in client.apply_batch_sync(batch, True):
            if msg.error:
                errors.add(msg.error)
            else:
                controllers.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        self.walkers = self.world.get_actors(walkers)
        self.controllers = self.world.get_actors(controllers)

        self.world.set_pedestrians_cross_factor(1)
        
        for controller in self.controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.4 + np.random.randn())

        self.timers = [np.random.randint(60, 600) * 20 for _ in self.controllers]

        print('%d / %d pedestrians spawned.' % (len(self.walkers), n_pedestrians))

    def _get_spawn_point(self, n_retry=10):
        for _ in range(n_retry):
            spawn = carla.Transform()
            spawn.location = self.world.get_random_location_from_navigation()

            if spawn.location is not None:
                return spawn

        raise ValueError('No valid spawns.')
    
    def check_dist(self, x, y):
        
        nearest = 1e5
        for walker in self.walkers:
            transform = walker.get_transform()
            p_x, p_y = transform.location.x, transform.location.y
            dist = math.sqrt((x-p_x)**2 + (y-p_y)**2)
            if dist < nearest:
                nearest = dist
        return nearest

    def tick(self):
        for i, controller in enumerate(self.controllers):
            self.timers[i] -= 1

            if self.timers[i] <= 0:
                self.timers[i] = np.random.randint(60, 600) * 20
                controller.go_to_location(self.world.get_random_location_from_navigation())

    def __del__(self):
        for controller in self.controllers:
            controller.stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers])


