from agents.algorithms.dqn import Agent
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from env.carla_env import CarlaEnv
import json
import numpy as np

if __name__ == "__main__":

    with open("config.json", 'r') as f:

        conf = json.load(f)
        env_conf = conf['carla']
        conf = conf['dqn']

    env_name = conf['env_name']

    run_name = f"{env_name}__{conf['exp_name']}_{int(time.time())}"
    writer = SummaryWriter(f"{conf['log_dir']}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in conf.items()])),
    )

    with CarlaEnv(env_conf) as env:
        screen_height = env.im_height
        screen_width = env.im_width
        frame_num = env.frame_num

        agent = Agent(
            gamma=conf['gamma'], epsilon=conf['epsilon'], batch_size=conf['batch_size'], n_action=env.n_action,
            eps_end=conf['eps_end'], input_dims=(frame_num, screen_height, screen_width), lr=conf['learning_rate'],
            eps_dec=conf['eps_dec'],
            max_mem_size=conf['max_mem_size'], use_per=conf['use_per'], num_classes=env.num_classes
        )
        n_time_steps = conf['n_time_steps']

        n_step = 0
        with tqdm(list(range(n_time_steps))) as tepoch:
            while n_step < n_time_steps:
                score = 0
                done = False
                observation = env.reset()
                speeds = []
                covered_dist = 0
                col_with_ped = 0
                while not done:

                    action = agent.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    tepoch.set_description(f"Step: {n_step}, reward: {reward}")
                    score += reward
                    agent.store_transition(observation, action, reward, observation_, done)
                    for _ in range(conf['n_step_update']):
                        agent.learn(n_step)
                    observation = observation_
                    speeds.append(sum(info['linear_speeds']) / len(info['linear_speeds']))
                    covered_dist = info['dist_covered']
                    col_with_ped = 1 if info['col_with_ped'] == 1 else col_with_ped
                    n_step += 1
                    tepoch.update(1)

                writer.add_scalar("charts/Episodic Return", score, n_step)
                writer.add_scalar("charts/Epsilon", agent.epsilon, n_step)
                writer.add_scalar("charts/Average Linear Velocity per Episode(km/h)", np.mean(speeds), n_step)
                writer.add_scalar("charts/Percentage of Covered Distance per Episode", covered_dist, n_step)
                writer.add_scalar("charts/Episode Terminated by Collision", col_with_ped, n_step)

    agent.save_models()
