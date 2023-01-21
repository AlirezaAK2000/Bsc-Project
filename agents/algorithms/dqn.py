import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from agents.common.buffer import ReplayBuffer
from agents.common.networks import DeepQNetwork

    

class Agent():
    
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size
                 ,n_action, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4) -> None:
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_action)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.Q_eval = DeepQNetwork(n_action=n_action, input_dims=input_dims,
                                  fc1_dims=256, fc2_dims=256).to(self.device)
        
        self.target_eval = DeepQNetwork(n_action=n_action, input_dims=input_dims,
                                  fc1_dims=256, fc2_dims=256).to(self.device)
        
        self.target_eval.load_state_dict(self.Q_eval.state_dict())
        self.target_eval.eval()
        
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(max_size=self.mem_size, input_shape=input_dims, n_actions=1)
        
        self.store_transition = self.replay_buffer.store_transition
        
        
    def save_models(self):
        print('... saving models ...')
        self.Q_eval.save_checkpoint()
        self.target_eval.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.Q_eval.load_checkpoint()
        self.target_eval.load_checkpoint()
    
        
    def choose_action(self, observation):
        
        if np.random.random() > self.epsilon:
            with T.no_grad():
                state = T.tensor([observation]).to(self.device)
                actions = self.Q_eval.forward(state)
                action = T.argmax(actions).item()
        
        else:
            action = np.random.choice(self.action_space)
            
        return action
    
    def learn(self, steps):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return
        
        self.optimizer.zero_grad()
        
        
        states, actions, rewards, states_, dones = self.replay_buffer.sample_buffer(self.batch_size)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(states, dtype=T.float).to(self.device)
        new_state_batch = T.tensor(states_, dtype=T.float).to(self.device)
        reward_batch = T.tensor(rewards, dtype=T.float).to(self.device)
        terminal_batch = T.tensor(dones).to(self.device)
        action_batch = actions.squeeze()
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.target_eval.forward(new_state_batch).detach()
        
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.loss(q_target, q_eval).to(self.device)
        loss.backward()
        for param in self.Q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min
        
        if steps % 5 == 0:
            self.target_eval.load_state_dict(self.Q_eval.state_dict())
            
        
        
        
        
        
        
        
        
        
            
        
        