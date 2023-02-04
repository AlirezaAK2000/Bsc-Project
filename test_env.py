from env.carla_env import CarlaEnv
import time

if __name__ == '__main__':
    
    
    episodes = 10
    for _ in range(episodes):
        print('====================================================================================')
        with CarlaEnv() as env: 
            env.reset()
            print("Episode started")
            episode_start = time.time()
            while True:
                print("#####################################3")
                step_start = time.time()
                
                state, reward, done, info = env.step(1)
                
                print(info)
                
                step_end = time.time()
                
                print(f"Step Time time : {round(step_end - step_start, 4)}")
                
                if done:
                    print('episode finished')
                    break
                
                    
            episode_end = time.time()
            print(f"Episode time : {round(episode_end - episode_start, 4)}")
        