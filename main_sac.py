import numpy as np
from agents.algorithms.sac import Agent
from torch.utils.tensorboard import SummaryWriter
from env.carla_env import CarlaEnv
import json
import time
from tqdm import tqdm

if __name__ == '__main__':

    with open("config.json", 'r') as f:

        conf = json.load(f)
        env_conf = conf['carla']
        conf = conf['sac']

    env_name = conf['env_name']

    run_name = f"{env_name}__{conf['exp_name']}_{int(time.time())}"
    writer = SummaryWriter(f"{conf['log_dir']}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in conf.items()])),
    )

    with CarlaEnv(env_conf, continuous_action=True) as env:
        n_steps = conf['n_steps']
        screen_height = env.im_height
        screen_width = env.im_width
        frame_num = env.frame_num
        num_updates = conf['num_updates']

        agent = Agent(action_space_max=[env.throttle_max, env.brake_max], alpha=conf['alpha'], beta=conf['beta'],
                      input_dims=(frame_num, screen_height, screen_width),
                      gamma=conf['gamma'], n_actions=env.action_dim, max_size=conf['memory_size'], tau=conf['tau'],
                      batch_size=conf['batch_size'],
                      reward_scale=conf['reward_scale'])

        load_checkpoint = False

        if load_checkpoint:
            agent.load_models()

        time_step = 0
        n_games = 0
        with tqdm(list(range(n_steps))) as tepoch:

            while time_step < n_steps:
                observation = env.reset()
                done = False
                score = 0
                speeds = []
                covered_dist = 0
                col_with_ped = 0
                while not done:
                    tepoch.set_description(f"Step: {time_step} score: {score}")

                    action = agent.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    score += reward
                    agent.remember(observation, action, reward, observation_, done)

                    observation = observation_
                    if time_step % 20 == 0:
                        for _ in range(num_updates):
                            agent.learn()

                    speeds.append(sum(info['linear_speeds']) / len(info['linear_speeds']))
                    covered_dist = info['dist_covered']
                    col_with_ped = 1 if info['col_with_ped'] == 1 else col_with_ped
                    time_step += 1
                    tepoch.update(1)

                writer.add_scalar("charts/Episodic Return", score, time_step)
                writer.add_scalar("charts/Average Linear Velocity per Episode(km/h)", np.mean(speeds), n_games)
                writer.add_scalar("charts/Percentage of Covered Distance per Episode", covered_dist, n_games)
                writer.add_scalar("charts/Episode Terminated by Collision", col_with_ped, n_games)
                n_games += 1

                if time_step % 100 == 0:
                    if not load_checkpoint:
                        agent.save_models()
