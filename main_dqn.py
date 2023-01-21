import gym
from agents.algorithms.dqn import Agent
from agents.common.utils import plot_learning_curve
import numpy as np
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import time
from distutils.util import strtobool


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    
    parser.add_argument("--eps_dec", type=float, default=1e-4,
        help="Epsilon decay")
    
    parser.add_argument("--batch_size", type=int, default=256,
        help="Batch size for each update")
    
    parser.add_argument("--max_mem_size", type=int, default=1_000_000,
        help="size of the replay buffer")
    
    parser.add_argument("--eps_end", type=float, default=0.01,
        help="Min value of epsilon")
    
    parser.add_argument("--gamma", type=float, default=0.99,
        help="Discount factor")
    
    parser.add_argument("--epsilon", type=float, default=1.0,
        help="Epsilon for exploration")
    
    parser.add_argument("--n_games", type=int, default=500,
        help="Number of episodes")
    
    parser.add_argument("--log_dir", type=str, default='runs/',
        help="Base dir of tensorboards logs")
    
    parser.add_argument("--use_per", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use a perioritized replay buffer")
    
    args = parser.parse_args()
    
    env_name = 'MountainCar-v0'
    
    env = gym.make(env_name)
    agent = Agent(
        gamma= args.gamma , epsilon=args.epsilon, batch_size=args.batch_size, n_action=env.action_space.n,
        eps_end= args.eps_end, input_dims=env.observation_space.shape, lr=args.learning_rate, eps_dec= args.eps_dec,
        max_mem_size=args.max_mem_size, use_pre=args.use_per
    )
    run_name = f"{env_name}__{args.exp_name}_{int(time.time())}"

    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    
    scores, eps_history = [], []
    n_games = args.n_games
    
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)            
            
            agent.learn(steps)
            observation = observation_
            steps += 1
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        
        writer.add_scalar("charts/Episodic Return", score, i)
        writer.add_scalar("charts/Epsilon", agent.epsilon, i)
        writer.add_scalar("charts/Average Score Over 100 Previous Episodes", avg_score, i)

        
        
        if i % 100 == 0:
            agent.save_models()
        
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    
    x = [i + 1 for i in range(n_games)]
    file_name = f'{env_name}-{args.exp_name}.png'
    plot_learning_curve(x, scores, file_name)
    