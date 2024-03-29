import numpy as np
import matplotlib.pyplot as plt
import torch
import math

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    
    

# The ‘sum-tree’ data structure used here is very similar in spirit to the array representation
# of a binary heap. However, instead of the usual heap property, the value of a parent node is
# the sum of its children. Leaf nodes store the transition priorities and the internal nodes are
# intermediate sums, with the parent node containing the sum over all priorities, p_total. This
# provides a efficient way of calculating the cumulative sum of priorities, allowing O(log N) updates
# and sampling. (Appendix B.2.1, Proportional prioritization)

# Additional useful links
# Good tutorial about SumTree data structure:  https://adventuresinmachinelearning.com/sumtree-introduction-python/
# How to represent full binary tree as array: https://stackoverflow.com/questions/8256222/binary-tree-represented-using-array
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"




############################################################################################################
######################################### State Prepration For Car Racing###################################
############################################################################################################

def prepare_state(state, screen_height, screen_width):
    width_margin = (state.shape[1] - screen_width) // 2
    height_margin = (state.shape[1] - screen_height) // 2
    state = state[height_margin:-height_margin,width_margin:-width_margin,:]
    state = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])[:,:,None]
    
    state = torch.tensor(state.transpose((2, 0, 1)), device= 'cpu')
    return torch.unsqueeze(state, 0)




def reset(env, action, screen_height, screen_width, plotter, frame_num):
    
    state = prepare_state(env.reset(), screen_height, screen_width)
    plotter.set_data(state.numpy().squeeze(axis=0).transpose((1, 2, 0)))
    plt.pause(0.0000001)
    frames = [state]
    for _ in range(1, frame_num):
        state, _, _, _ = env.step(action)
        state = prepare_state(state, screen_height, screen_width)
        plotter.set_data(state.numpy().squeeze(axis=0).transpose((1, 2, 0)))
        frames.append(state/255.0)
        plt.pause(0.0000001)
        
    frames = torch.cat(frames, dim=1)
    
    return frames


            
        
        
def step(env, action, screen_height, screen_width, plotter, frame_num, reward_memory):
    rewards = 0
    frames = []
    done = False
    for _ in range(frame_num):
        if not done:
            state, reward, done, info = env.step(action)
            rewards += np.clip(reward, a_max=1.0, a_min=-math.inf)
            done = True if reward_memory(reward) <= -0.1 else False
                
            state = prepare_state(state, screen_height, screen_width)
            plotter.set_data(state.numpy().squeeze(axis=0).transpose((1, 2, 0)))
            frames.append(state/255.0)
            plt.pause(0.0000001)
        else:
            state = prepare_state(np.zeros((96,96,3)), screen_height, screen_width)
            frames.append(state)
    
    
    frames = torch.cat(frames, dim=1)
    
    return frames, done, rewards

def reward_memory():
    # record reward for last 100 steps
    count = 0
    length = 80
    history = np.zeros(length)

    def memory(reward):
        nonlocal count
        history[count] = reward
        count = (count + 1) % length
        return np.mean(history)
    
    return memory