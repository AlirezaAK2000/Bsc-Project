import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
from agents.algorithms.ppo import PPO
import json
from env.carla_env import CarlaEnv
from tqdm import tqdm
import numpy as np


def train():
    with open("config.json", 'r') as f:

        conf = json.load(f)
        env_conf = conf['carla']
        conf = conf['ppo']

    env_name = "carla"

    has_continuous_action_space = conf['has_continuous_action_space']

    max_training_timesteps = conf["max_training_timesteps"]

    save_model_freq = conf["save_model_freq"]  # save model frequency (in num timesteps)

    action_std = conf['action_std']  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = conf[
        'action_std_decay_rate']  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = conf["min_action_std"]  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = conf['action_std_decay_freq']  # action_std decay frequency (in num timesteps)

    K_epochs = conf["k_epochs"]  # update policy for K epochs in one PPO update

    eps_clip = conf["eps_clip"]  # clip parameter for PPO
    gamma = conf['gamma']  # discount factor

    lr_actor = conf['lr_actor']  # learning rate for actor network
    lr_critic = conf['lr_critic']  # learning rate for critic network
    update_timestep = conf['update_timestep']

    random_seed = 0  # set random seed if required (0 = no random seed)

    with CarlaEnv(env_conf, has_continuous_action_space) as env:

        if has_continuous_action_space:
            action_dim = env.action_dim
        else:
            action_dim = env.n_action

        ###################### logging ######################
        if not os.path.exists(conf["log_dir"]):
            os.makedirs(conf["log_dir"])
        run_name = f"{env_name}__{conf['exp_name']}_{int(time.time())}"

        writer = SummaryWriter(f"{conf['log_dir']}/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in conf.items()])),
        )

        ################### checkpointing ###################
        run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

        directory = "tmp"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + f"{env_name}_ppo" + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

        # initialize a PPO agent
        ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                        conf["ent_coe"],env.num_classes , action_std_init=action_std)

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # printing and logging variables
        time_step = 0
        i_episode = 0

        # training loop
        with tqdm(list(range(max_training_timesteps))) as tepoch:
            while time_step <= max_training_timesteps:

                state = env.reset()
                current_ep_reward = 0
                done = False
                speeds = []
                covered_dist = 0
                col_with_ped = 0

                while not done:
                    # select action with policy
                    action = ppo_agent.select_action(state)
                    state, reward, done, info = env.step(action)
                    tepoch.set_description(f"Step: {time_step}, reward: {reward}")

                    # saving reward and is_terminals
                    ppo_agent.buffer.rewards.append(reward * conf['reward_scale'])
                    ppo_agent.buffer.is_terminals.append(done)

                    current_ep_reward += reward

                    # update PPO agent
                    if (time_step + 1) % update_timestep == 0:
                        ppo_agent.update()

                    # if continuous action space; then decay action std of ouput action distribution
                    if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                    speeds.append(sum(info['linear_speeds']) / len(info['linear_speeds']))
                    covered_dist = info['dist_covered']
                    col_with_ped = 1 if info['col_with_ped'] == 1 else col_with_ped

                    # save model weights
                    if (time_step + 1) % save_model_freq == 0:
                        print("model saved")
                        ppo_agent.save(checkpoint_path)
                    time_step += 1
                    tepoch.update(1)
                    # break; if the episode is over
                i_episode += 1
                writer.add_scalar("charts/Episodic Return", current_ep_reward, time_step)
                writer.add_scalar("charts/Average Linear Velocity per Episode(km/h)", np.mean(speeds), time_step)
                writer.add_scalar("charts/Percentage of Covered Distance per Episode", covered_dist, time_step)
                writer.add_scalar("charts/Episode Terminated by Collision", col_with_ped, time_step)

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")


if __name__ == '__main__':
    train()
