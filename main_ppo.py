import gym
import numpy as np
from agents.algorithms.ppo import Agent
from agents.common.utils import plot_learning_curve
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import time

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    
    parser.add_argument("--learning_rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    
    parser.add_argument("--batch_size", type=int, default=25,
        help="Batch size for each update")
    
    parser.add_argument("--n_games", type=int, default=500,
        help="Number of episodes")

    parser.add_argument("--n_epochs", type=int, default=5,
        help="Number of Epochs for Updating the networks")
    
    parser.add_argument("--log_dir", type=str, default='runs/',
        help="Base dir of tensorboards logs")

    
    
    args = parser.parse_args()
    
    env_name = 'MountainCar-v0'
    
    env = gym.make(env_name)
    
    run_name = f"{env_name}__{args.exp_name}_{int(time.time())}"

    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    batch_size = args.batch_size
    N = batch_size * 4
    n_epochs = args.n_epochs
    alpha = args.learning_rate
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape, mem_size = N)
    n_games = args.n_games

    figure_file = f'{env_name}-ppo.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        writer.add_scalar("charts/Episodic Return", score, i)
        writer.add_scalar("charts/Average Score Over 100 Previous Episodes", avg_score, i)


        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
