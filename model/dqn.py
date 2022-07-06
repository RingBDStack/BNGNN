import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as F
from copy import deepcopy
import random

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
class Memory(object):
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
    def save(self, state, action, reward, next_state):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state)
        self.memory.append(transition)
    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

class QAgent(object):
    def __init__(self,
                 replay_memory_size, replay_memory_init_size, update_target_estimator_every, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps,
                 lr, batch_size, num_net, action_num, norm_step, mlp_layers, state_shape, device):
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
        self.train_t = 0
        self.q_estimator = Estimator(action_num=action_num, lr=lr, state_shape=self.state_shape, mlp_layers=mlp_layers, device=device)
        self.target_estimator = Estimator(action_num=action_num, lr=lr, state_shape=self.state_shape, mlp_layers=mlp_layers, device=self.device)
        self.memory = Memory(replay_memory_size, batch_size)
    def learn(self, env, total_timesteps):
        env.best_policy = deepcopy(self)
        last_val = 0.0
        next_state = env.reset()
        index = env.train_idx[0]
        for t in range(total_timesteps):
            best_action, epsilon = self.predict_batch(next_state, t)
            exploration_flag = np.random.choice([True, False], p=[epsilon, 1-epsilon], size=1)
            if exploration_flag:
                best_action = np.random.choice([0,1,2], 1)[0]
            state = next_state
            next_state, index, reward, val_acc = env.step(best_action, index)
            self.memory.save(state, best_action, reward, next_state)
            if t > self.batch_size:
                self.train()
            if val_acc > last_val:
                env.best_policy = deepcopy(self)
                last_val = val_acc
    def eval_step(self, states):
        q_values = self.q_estimator.predict_nograd(states)
        best_actions = np.argmax(q_values, axis=-1)
        return best_actions
    def predict_batch(self, state, t):
        epsilon = self.epsilons[min(t, self.epsilon_decay_steps-1)]
        q_values = self.q_estimator.predict_nograd(state)
        best_action = np.argmax(q_values, axis=1)[0]
        return best_action, epsilon
    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample()
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        best_actions_next = np.argmax(q_values_next_target, axis=1)
        target_batch = reward_batch + self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions_next]
        self.q_estimator.update(state_batch, action_batch, target_batch)
        self.target_estimator = deepcopy(self.q_estimator)

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
            if len(states.shape) < 2:
                states = states.unsqueeze(0)
            q_values = F.softmax(self.qnet(states), dim=1).to('cpu').numpy()
        return q_values
    def update(self, s, a, y):
        self.optimizer.zero_grad()
        self.qnet.train()
        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)
        q_values = self.qnet(s)
        Q = torch.gather(q_values, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)
        Q_loss = self.mse_loss(Q, y)
        Q_loss.backward()
        self.optimizer.step()
        self.qnet.eval()

class EstimatorNetwork(nn.Module):
    def __init__(self, action_num, state_shape, mlp_layers):
        super(EstimatorNetwork, self).__init__()
        layer_dims = [state_shape[-1]] + mlp_layers
        fc = [nn.Flatten()]
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], action_num, bias=True))
        self.fc_layers = nn.Sequential(*fc)
    def forward(self, state):
        return self.fc_layers(state)
