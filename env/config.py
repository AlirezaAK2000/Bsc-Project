import numpy as np
import carla
import json


PRESET_WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    # 2: carla.WeatherParameters.CloudyNoon,
    # 3: carla.WeatherParameters.WetNoon,
    # 5: carla.WeatherParameters.MidRainyNoon,
    # 4: carla.WeatherParameters.WetCloudyNoon,
    # 6: carla.WeatherParameters.HardRainNoon,
    # 7: carla.WeatherParameters.SoftRainNoon,

    # 8: carla.WeatherParameters.ClearSunset,
    # 9: carla.WeatherParameters.CloudySunset,
    # 10: carla.WeatherParameters.WetSunset,
    # 12: carla.WeatherParameters.MidRainSunset,
    # 11: carla.WeatherParameters.WetCloudySunset,
    # 13: carla.WeatherParameters.HardRainSunset,
    # 14: carla.WeatherParameters.SoftRainSunset,
}

WEATHERS = list(PRESET_WEATHERS.values())


base_classes = {
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

main_classes = {
    'Other': 0,  
    'Building': 1,  
    'RoadLine': 2,  
    'Road': 3,  
    'SideWalk': 4,  
    'Vehicles': 5,  
    'Wall': 6,  
    'Ground': 7,  
    'RailTrack': 8,  
    'Dynamic': 9,  
    'Pedestrian': 10  
}
