import carla


if __name__ == '__main__':
    
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    
    client.show_recorder_file_info("/home/alirezaak/.config/Epic/CarlaUE4/Saved/log.log")
        
        