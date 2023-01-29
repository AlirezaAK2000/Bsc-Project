import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import time
from agents.algorithms.ppo import PPO
import argparse
import matplotlib.pyplot as plt
from agents.common.utils import reset, step, reward_memory

################################### Training ###################################
def train():
    print("============================================================================================")
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    
    parser.add_argument("--lr_actor", type=float, default=0.0003,
        help="the learning rate of the actor optimizer")
    
    parser.add_argument("--lr_critic", type=float, default=0.001,
        help="the learning rate of the critic optimizer")
    
    parser.add_argument("--k_epochs", type=int, default=15,
        help="Number of Epochs for Updating the networks")
    
    parser.add_argument("--max_ep_len", type=int, default=200,
        help="Maximum time steps of the episodes")
    
    parser.add_argument("--max_training_timesteps", type=int, default=5e5,
        help="number of time steps.")
    
    parser.add_argument("--log_dir", type=str, default='runs/',
        help="Base dir of tensorboards logs")
    
    parser.add_argument("--action_std", type=float, default=0.8,
        help="Maximum Std for continuous setting")
    
    parser.add_argument("--action_std_decay_rate", type=float, default=0.05,
        help="Std decay rate for continuous setting")
    
    parser.add_argument("--min_action_std", type=float, default=0.1,
        help="Min Std for continuous setting")
    
    parser.add_argument("--action_std_decay_freq", type=int, default=2.5e3,
        help="Maximum time steps of the episodes")
    
    args = parser.parse_args()


    ####### initialize environment hyperparameters ######
    use_conv = True
    env_name = "CarRacing-v2"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = args.max_ep_len                   # max timesteps in one episode
    max_training_timesteps = int(args.max_training_timesteps)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len        # print avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)

    action_std = args.action_std                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = args.action_std_decay_rate         # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = args.min_action_std                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = args.action_std_decay_freq  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len // 4      # update policy every n timesteps
    K_epochs = args.k_epochs               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = args.lr_actor       # learning rate for actor network
    lr_critic = args.lr_critic       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    
    # for image data 
    screen_height = 72
    screen_width = 84
    frame_num = 4
    #####################################################

    print("training environment name : " + env_name)

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    run_name = f"{env_name}__{args.exp_name}_{int(time.time())}"
    
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 1

    # log_running_reward = 0
    # log_running_episodes = 0

    time_step = 0
    i_episode = 0

    ax = plt.subplot(111)
    prog_shower = ax.imshow(np.zeros((screen_height, screen_width, 1)) , cmap='gray', vmin=0, vmax=255)
    plt.ion()

    # training loop
    while time_step <= max_training_timesteps:

        state = reset(env, [0,1,0.8], screen_height, screen_width, prog_shower, frame_num)
        current_ep_reward = 0
        rew_mem = reward_memory()
        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, done,reward = step(env, action, screen_height, screen_width,
                                       prog_shower, frame_num, rew_mem)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                print("policy updated")

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)


            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1
        
        writer.add_scalar("charts/Episodic Return", current_ep_reward, time_step)

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
    
    
    
    
    
    
    