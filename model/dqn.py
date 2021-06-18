import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import random

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
class Normalizer(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.state_memory = []
        self.max_size = 1000
        self.length = 0

    def normalize(self, s):
        if self.length == 0:
            return s
        return (s - self.mean) / (self.std + 1e-8)

    def append(self, s):
        if len(self.state_memory) > self.max_size:
            self.state_memory.pop(0)
        self.state_memory.append(s)
        self.mean = np.mean(self.state_memory, axis=0)
        self.std = np.std(self.state_memory, axis=0)
        self.length = len(self.state_memory)


class Memory(object):
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

class QAgent(object):
    def __init__(self,
                 replay_memory_size, replay_memory_init_size, update_target_estimator_every,
                 discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps,
                 lr, batch_size,
                 num_net,
                 action_num,
                 norm_step,
                 mlp_layers,
                 state_shape,
                 device):
        self.replay_memory_size = replay_memory_size
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.num_net = num_net
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.action_num = action_num
        self.norm_step = norm_step
        self.device = device
        self.total_t = 0
        self.train_t = 0

        self.q_estimator = Estimator(action_num=action_num,
                                     lr=lr,
                                     state_shape=self.state_shape,
                                     mlp_layers=mlp_layers,
                                     device=device)

        self.target_estimator = Estimator(action_num=action_num,
                                          lr=lr,
                                          state_shape=self.state_shape,
                                          mlp_layers=mlp_layers,
                                          device=self.device)
        self.memory = Memory(replay_memory_size, batch_size)
        self.normalizer = Normalizer()

    def learn(self, env, total_timesteps):
        next_states = env.reset()
        trajectories = []
        for t in range(total_timesteps):
            A = self.predict_batch(next_states, t)
            best_actions = np.random.choice(np.arange(len(A)), p=A, size=next_states.shape[0])
            states = next_states
            next_states, rewards, dones, debug = env.step(best_actions)
            trajectories = zip(states, best_actions, rewards, next_states, dones)
            for each in trajectories:
                self.feed(each)
        loss = self.train()
        return loss, rewards, debug

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        if self.total_t < self.norm_step:
            self.feed_norm(state)
        else:
            self.feed_memory(state, action, reward, next_state, done)
        self.total_t += 1

    def eval_step(self, states):
        q_values = self.q_estimator.predict_nograd(states)
        best_actions = np.argmax(q_values, axis=-1)
        return best_actions

    def predict_batch(self, states, t):
        epsilon = self.epsilons[min(t, self.epsilon_decay_steps-1)]
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        q_values = self.q_estimator.predict_nograd(self.normalizer.normalize(states))
        best_action = np.argmax(q_values, axis=1)
        for a in best_action:
            A[best_action] += (1.0 - epsilon)
        A = A/A.sum()
        return A

    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)

        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]
        state_batch = np.array(state_batch)
        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
        self.train_t += 1
        return loss

    def feed_norm(self, state):
        self.normalizer.append(state)

    def feed_memory(self, state, action, reward, next_state,done):
        self.memory.save(self.normalizer.normalize(state), action, reward, self.normalizer.normalize(next_state), done)


class Estimator(object):
    def __init__(self,
                 action_num,
                 lr,
                 state_shape,
                 mlp_layers,
                 device):
        self.device = device
        qnet = EstimatorNetwork(action_num, state_shape, mlp_layers)
        self.qnet = qnet.to(device)
        self.qnet.eval()
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=lr)

    def predict_nograd(self, states):
        with torch.no_grad():
            states = torch.from_numpy(states).float().to(self.device)
            q_as = self.qnet(states).to('cpu').numpy()
        return q_as

    def update(self, s, a, y):
        self.optimizer.zero_grad()
        self.qnet.train()
        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)
        q_as = self.qnet(s)
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
        Q_loss = self.mse_loss(Q, y)
        Q_loss.backward()
        self.optimizer.step()
        Q_loss = Q_loss.item()
        self.qnet.eval()
        return Q_loss

class EstimatorNetwork(nn.Module):
    def __init__(self,
                 action_num,
                 state_shape,
                 mlp_layers):
        super(EstimatorNetwork, self).__init__()
        layer_dims = [state_shape[-1]] + mlp_layers
        fc = [nn.Flatten()]
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], action_num, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, states):
        return self.fc_layers(states)
