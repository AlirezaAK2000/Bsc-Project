import gym
from agents.algorithms.dqn import Agent
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import time
from distutils.util import strtobool
from tqdm import tqdm
import matplotlib.pyplot as plt
# from agents.common.utils import reset, step, reward_memory
from env.carla_env import CarlaEnv
import json
import numpy as np

if __name__ == "__main__":
    
    with open("config.json", 'r') as f:
        
        conf = json.load(f)
        env_conf = conf['carla']
        conf = conf['dqn']
    
    screen_height = conf['screen_height']
    screen_width = conf['screen_width']
    frame_num = conf['frame_num']
    
    env_name = conf['env_name']
    
    run_name = f"{env_name}__{conf['exp_name']}_{int(time.time())}"
    writer = SummaryWriter(f"{conf['log_dir']}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in conf.items()])),
    )

    with CarlaEnv(env_conf) as env:
        
        agent = Agent(
            gamma= conf['gamma'] , epsilon=conf['epsilon'], batch_size=conf['batch_size'], n_action=env.n_action,
            eps_end= conf['eps_end'], input_dims=(frame_num, screen_height, screen_width), lr=conf['learning_rate'], eps_dec= conf['eps_dec'],
            max_mem_size=conf['max_mem_size'], use_per=conf['use_per']
        )
        scores, eps_history = [], []
        n_time_steps = conf['n_time_steps']
        
        
        n_games = 0
        n_step = 0
        with tqdm(list(range(n_time_steps))) as tepoch:
            while n_step < n_time_steps:
                score = 0
                done = False
                observation = env.reset()
                speeds = []
                dists = []
                while not done:

                    tepoch.set_description(f"Step: {n_step}")
                    action = agent.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    score += reward
                    agent.store_transition(observation, action, reward, observation_, done)            
                    agent.learn(n_step)
                    observation = observation_
                    
                    if done:
                        
                        scores.append(score)
                        eps_history.append(agent.epsilon)
                        avg_score = np.mean(scores[-100:])
                        
                        writer.add_scalar("charts/Episodic Return", score, n_step)
                        writer.add_scalar("charts/Epsilon", agent.epsilon, n_step)
                        writer.add_scalar("charts/Average Score Over 100 Previous Episodes", avg_score, n_step)
                        
                    n_step += 1
                    speeds.append(sum(info['linear_speeds']) / len(info['linear_speeds']))
                    p, p_ = info['locs'][0], info['locs'][-1]
                    dists.append(math.sqrt((p[0] - p_[0])**2 + (p[1] - p_[1])**2))
                    
                    
                    tepoch.update(1)
                
                writer.add_scalar("charts/Average Linear Velocity per Episode(km/h)", (speeds), n_games)
                writer.add_scalar("charts/Covered Distance per Episode(m)", avg_score, n_games)
                n_games += 1
            
            if i % 100 == 0:
                agent.save_models()
            
            print('Step ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        
        x = [i + 1 for i in range(n_games)]
        