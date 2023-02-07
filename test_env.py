from env.carla_env import *
import time
import json

if __name__ == '__main__':
    episodes = 3
    with open("config.json", 'r') as f:

        conf = json.load(f)
        conf = conf['carla']

    with CarlaEnv(conf) as env:
        for _ in range(episodes):
            print('====================================================================================')
            env.reset()
            print("Episode started")
            episode_start = time.time()
            # brake_step = 0
            while True:
                print("######################################")
                step_start = time.time()

                # if brake_step < 10:
                state, reward, done, info = env.step(STRAIGHT)
                # brake_step += 1
                # else:
                #     state, reward, done, info = env.step(BRAKE)
                #     brake_step = 0

                print(info['linear_speeds'])
                print(f"reward: {reward}")

                step_end = time.time()

                print(f"Step Time time : {round(step_end - step_start, 4)}")

                if done:
                    print('episode finished')
                    break

            episode_end = time.time()
            print(f"Episode time : {round(episode_end - episode_start, 4)}")
