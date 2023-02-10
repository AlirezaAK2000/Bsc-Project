import gym
import numpy as np
from agents.algorithms.sac import Agent
from agents.common.utils import plot_learning_curve
from torch.utils.tensorboard import SummaryWriter
from agents.common.utils import step, reward_memory, reset
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env_name = 'CarRacing-v2'
    n_games = 500
    screen_height = 72
    screen_width = 84
    frame_num = 4
    num_updates = 20

    env = gym.make(env_name, continuous=True)
    agent = Agent(input_dims=(frame_num, screen_height, screen_width), env=env,
                  n_actions=env.action_space.shape[0])
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = f'{env_name}-sac.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')



    for i in range(n_games):
        observation = reset(env, [0, 1, 0.8], screen_height, screen_width, prog_shower, frame_num)
        done = False
        score = 0
        rew_mem = reward_memory()
        while not done:
            action = agent.choose_action(observation)
            observation_, done, reward = step(env, action, screen_height, screen_width,
                                              prog_shower, frame_num, rew_mem)
            score += reward
            agent.remember(observation, action, reward, observation_, done)

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        for _ in range(num_updates):
            agent.learn()

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
