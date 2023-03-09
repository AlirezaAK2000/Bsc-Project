import torch as T
import torch.optim as optim
import numpy as np
from agents.common.networks import DeepQNetworkAtari
from agents.common.utils import SumTree
import random


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, num_classes):
        self.mem_size = max_size
        self.count = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.num_classes = num_classes

    def store_transition(self, state, action, reward, state_, done):
        index = self.count % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.count, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch] / self.num_classes
        states_ = self.new_state_memory[batch] / self.num_classes
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, num_classes, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=max_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        self.size = max_size
        self.num_classes = num_classes

        self.state_memory = np.zeros((self.size, *input_shape))
        self.new_state_memory = np.zeros((self.size, *input_shape))
        self.action_memory = np.zeros((self.size, n_actions))
        self.reward_memory = np.zeros(self.size)
        self.terminal_memory = np.zeros(self.size, dtype=np.bool)

        self.count = 0
        self.real_size = 0

    def store_transition(self, state, action, reward, state_, done):

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        self.state_memory[self.count] = state
        self.action_memory[self.count] = action
        self.reward_memory[self.count] = reward
        self.new_state_memory[self.count] = state_
        self.terminal_memory[self.count] = done

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample_buffer(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = np.zeros((batch_size, 1))

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = (
            self.state_memory[sample_idxs] / self.num_classes,
            self.action_memory[sample_idxs],
            self.reward_memory[sample_idxs],
            self.new_state_memory[sample_idxs] / self.num_classes,
            self.terminal_memory[sample_idxs]
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, T.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class Agent:

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, num_classes
                 , n_action, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, use_per=True,
                 use_conv=True) -> None:

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

        self.Q_eval = DeepQNetworkAtari(n_action=n_action).to(self.device)

        self.target_eval = DeepQNetworkAtari(n_action=n_action).to(self.device)

        self.target_eval.load_state_dict(self.Q_eval.state_dict())
        self.target_eval.eval()

        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.use_per = use_per
        self.replay_buffer = PrioritizedReplayBuffer(max_size=self.mem_size, input_shape=input_dims,
                                                     n_actions=1, num_classes=num_classes) if self.use_per else \
            ReplayBuffer(max_size=self.mem_size, input_shape=input_dims, n_actions=1, num_classes=num_classes)

        self.store_transition = self.replay_buffer.store_transition
        print(f'using PER {self.use_per}')

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
                state = T.tensor(observation, dtype=T.float).to(self.device)
                actions = self.Q_eval.forward(state)
                action = T.argmax(actions).item()

        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, steps):
        if self.replay_buffer.count < self.batch_size:
            return

        self.optimizer.zero_grad()
        weights = None
        if self.use_per:
            b, weights, tree_idxs = self.replay_buffer.sample_buffer(self.batch_size)
            states, actions, rewards, states_, dones = b
            weights = T.tensor(weights, dtype=T.float)
        else:
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

        td_error = T.abs(q_target - q_eval).detach().cpu().numpy()

        if weights is None:
            weights = T.ones_like(q_eval)

        weights = weights.to(self.device)
        loss = T.mean((q_eval - q_target) ** 2 * weights)

        loss.backward()

        for param in self.Q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

        if steps % 5 == 0:
            self.target_eval.load_state_dict(self.Q_eval.state_dict())

        if self.use_per:
            self.replay_buffer.update_priorities(tree_idxs, td_error)
