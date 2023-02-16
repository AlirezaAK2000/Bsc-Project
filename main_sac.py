import numpy as np
import torch
from agents.algorithms.sac import ReplayBuffer
import random
from agents.algorithms.sac import SAC
from env.carla_env import CarlaEnv
import json
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm


def save(conf, save_name, model, ep=None):
    import os
    save_dir = './tmp/sac'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + conf['exp_name'] + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + conf['exp_name'] + save_name + ".pth")


def get_config():
    with open("config.json", 'r') as f:
        conf = json.load(f)

    return conf


def train(conf):
    env_conf = conf['carla']
    conf = conf['sac']
    np.random.seed(conf['seed'])
    random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    env_name = conf['env']
    run_name = f"{env_name}__{conf['exp_name']}_{int(time.time())}"
    writer = SummaryWriter(f"{conf['log_dir']}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in conf.items()])),
    )
    with CarlaEnv(env_conf) as env:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        agent = SAC(state_size=(env.frame_num, env.im_height, env.im_width),
                    action_size=env.action_dim,
                    device=device,
                    gamma=conf['gamma'],
                    tau=conf['tau'],
                    learning_rate=conf['learning_rate'])

        buffer = ReplayBuffer(buffer_size=conf['buffer_size'], batch_size=conf['batch_size'],
                              num_classes=env.num_classes, device=device)

        n_step = 0
        n_games = 0
        with tqdm(list(range(conf['n_steps']))) as tepoch:

            while n_step < conf['n_steps']:

                state = env.reset()
                score = 0
                done = False
                speeds = []
                covered_dist = 0
                col_with_ped = 0
                while not done:

                    action = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)
                    tepoch.set_description(f"Step: {n_step} reward: {reward}")

                    buffer.add(state, action, reward, next_state, done)
                    if len(buffer) >= conf['batch_size']:
                        agent.learn(buffer.sample())
                    state = next_state
                    score += reward
                    speeds.append(sum(info['linear_speeds']) / len(info['linear_speeds']))
                    covered_dist = info['dist_covered']
                    col_with_ped = 1 if info['col_with_ped'] == 1 else col_with_ped

                    tepoch.update(1)
                    if (n_step + 1) % conf['save_every'] == 0:
                        save(conf, save_name="SAC_discrete", model=agent.actor_local, ep=0)
                    n_step += 1

                writer.add_scalar("charts/Episodic Return", score, n_step)
                writer.add_scalar("charts/Average Linear Velocity per Episode(km/h)", np.mean(speeds), n_step)
                writer.add_scalar("charts/Percentage of Covered Distance per Episode", covered_dist, n_step)
                writer.add_scalar("charts/Episode Terminated by Collision", col_with_ped, n_step)
                n_games += 1


if __name__ == "__main__":
    config = get_config()
    train(config)
