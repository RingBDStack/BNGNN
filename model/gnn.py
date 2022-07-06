import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class GCN(nn.Module):
	def __init__(self,features, adjs, hid_dim, out_dim, drop, slope):
		super(GCN, self).__init__()
		self.features = torch.FloatTensor(features)
		self.features = nn.Parameter(self.features, requires_grad=False)
		self.adjs = torch.FloatTensor(adjs)
		self.adjs = self.adj_process(self.adjs)
		self.adjs = nn.Parameter(self.adjs, requires_grad=False)
		_, _, self.dim_in = features.shape
		self.fc1 = nn.Linear(self.dim_in, hid_dim, bias=False)
		self.fc2 = nn.Linear(hid_dim, hid_dim, bias=False)
		self.fc3 = nn.Linear(hid_dim, hid_dim, bias=False)
		self.classifier = nn.Linear(hid_dim, out_dim, bias=True)
		self.leaky_relu = nn.LeakyReLU(slope)
		self.dropout = nn.Dropout(drop)
		self.loss_function = nn.CrossEntropyLoss()
	def adj_process(self, adjs):
		net_num, n_num, n_num = adjs.shape
		adjs = adjs.detach()
		for i in range(net_num):
			adjs[i] += torch.eye(n_num)
			adjs[i][adjs[i]>0.] = 1.
			degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)
			degree_matrix = torch.pow(degree_matrix, -1/2)
			degree_matrix[degree_matrix == float("inf")] = 0.
			degree_matrix = torch.diag(degree_matrix)
			adjs[i] = torch.mm(degree_matrix, adjs[i])
			adjs[i] = torch.mm(adjs[i],degree_matrix)
		return adjs
	def forward(self, input):
		action, index = input
		features = self.features[index]
		adj = self.adjs[index]
		features = self.fc1(torch.matmul(adj, features))
		features = self.leaky_relu(features)
		features = self.dropout(features)
		if action == 1:
			features = self.fc2(torch.matmul(adj, features))
			features = self.leaky_relu(features)
			features = self.dropout(features)
		elif action == 2:
			features = self.fc2(torch.matmul(adj, features))
			features = self.leaky_relu(features)
			features = self.dropout(features)
			features = self.fc3(torch.matmul(adj, features))
			features = self.leaky_relu(features)
			features = self.dropout(features)
		if len(features.shape) < 3:
			predict = torch.mean(features,dim=0).unsqueeze(0)
		else:
			predict = torch.mean(features, dim=1)
		predict = self.classifier(predict)
		predict = F.log_softmax(predict, dim=1)
		return predict

class gnn_env(object):
	def __init__(self, dataset, view, max_layer, hid_dim, out_dim, drop, slope, lr, weight_decay, gnn_type, device, policy, benchmark_num):
		self.dataset = dataset
		self.view = view
		self.max_layer = max_layer
		self.action_num = max_layer
		self.device = device
		self.policy = policy
		self.benchmark_num = benchmark_num
		self.load_dataset()
		if gnn_type == 'GCN':
			self.model = GCN(self.init_net_feat, self.net_brain_adj, hid_dim, out_dim, drop, slope).to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
		self.batch_size_qdn = self.num_train
		self.state_shape = self.reset().shape
		self.past_performance = [0.]
	def load_dataset(self):
		if self.dataset in ['BP', 'HIV']:
			ratio = [56, 7, 7] if self.dataset == 'HIV' else [77, 10, 10]
			net_label, net_weighted, net_brain_adj, net_subject_adj = self.load_BP_HIV()
		elif self.dataset in ['ADHD', 'HI', 'GD']:
			if self.dataset == 'ADHD':
				ratio = [67, 8, 8]
			elif self.dataset == 'HI':
				ratio = [63, 8, 8]
			else:
				ratio = [69, 8, 8]
			net_label, net_weighted, net_brain_adj, net_subject_adj = self.load_ADHD_HI_GD()
		else:
			ratio = [49, 6, 6]
			net_label, net_weighted, net_brain_adj, net_subject_adj = self.load_HA()
		self.num_net = len(net_label)
		self.net_label = net_label
		all_idx = [i for i in range(self.num_net)]
		random.shuffle(all_idx)
		train_idx = all_idx[:ratio[0]]
		val_idx = all_idx[ratio[0]:ratio[0] + ratio[1]]
		test_idx = all_idx[ratio[0] + ratio[1]:]
		self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
		self.num_train, self.num_val, self.num_test = len(self.train_idx), len(self.val_idx), len(self.test_idx)
		self.init_net_feat = net_weighted
		self.net_brain_adj = net_brain_adj
		self.transition_adj = []
		self.transition_adj.append(net_subject_adj)
		tmp_subject_adj = net_subject_adj
		for _ in range(1, self.action_num):
			net_subject_adj = np.matmul(net_subject_adj, tmp_subject_adj)
			net_subject_adj[net_subject_adj>0] = 1
			self.transition_adj.append(net_subject_adj)
	def load_ADHD_HI_GD(self):
		net_label = np.load('./data/preprocessed/' + self.dataset + '/' + self.dataset + '_fMRI_label.npy')
		net_weighted = np.load('./data/preprocessed/' + self.dataset + '/' + self.dataset + '_fMRI_weighted.npy')
		net_brain_adj = np.load('./data/preprocessed/' + self.dataset + '/' + self.dataset + '_fMRI_adj_brain.npy')
		net_subject_adj = np.load('./data/preprocessed/' + self.dataset + '/' + self.dataset + '_fMRI_adj_subject.npy')
		return net_label, net_weighted, net_brain_adj, net_subject_adj
	def load_BP_HIV(self):
		if self.view == 'DTI':
			net_label = np.load('./data/preprocessed/' + self.dataset + '-DTI/' + self.dataset + '_DTI_label.npy')
			net_weighted = np.load('./data/preprocessed/' + self.dataset + '-DTI/' + self.dataset + '_DTI_weighted.npy')
			net_brain_adj = np.load('./data/preprocessed/' + self.dataset + '-DTI/' + self.dataset + '_DTI_adj_brain.npy')
			net_subject_adj = np.load('./data/preprocessed/' + self.dataset + '-DTI/' + self.dataset + '_DTI_adj_subject.npy')
		else:
			net_label = np.load('./data/preprocessed/' + self.dataset + '-fMRI/' + self.dataset + '_fMRI_label.npy')
			net_weighted = np.load('./data/preprocessed/' + self.dataset + '-fMRI/' + self.dataset + '_fMRI_weighted.npy')
			net_brain_adj = np.load('./data/preprocessed/' + self.dataset + '-fMRI/' + self.dataset + '_fMRI_adj_brain.npy')
			net_subject_adj = np.load('./data/preprocessed/' + self.dataset + '-fMRI/' + self.dataset + '_fMRI_adj_subject.npy')
		return net_label, net_weighted, net_brain_adj, net_subject_adj
	def load_HA(self):
		net_label = np.load('./data/preprocessed/' + self.dataset + '/' + self.dataset + '_EEG_label.npy')
		net_weighted = np.load('./data/preprocessed/' + self.dataset + '/' + self.dataset + '_EEG_weighted.npy')
		net_brain_adj = np.load('./data/preprocessed/' + self.dataset + '/' + self.dataset + '_EEG_adj_brain.npy')
		net_subject_adj = np.load('./data/preprocessed/' + self.dataset + '/' + self.dataset + '_EEG_adj_subject.npy')
		return net_label, net_weighted, net_brain_adj, net_subject_adj
	def reset(self):
		state = np.mean(self.net_brain_adj,axis=1)
		state = state[self.train_idx[0]]
		self.optimizer.zero_grad()
		return state
	def transition(self, action, index):
		neighbors = np.nonzero(self.transition_adj[action][index])[0]
		legal_neighbors = np.array(self.train_idx)
		neighbors = np.intersect1d(neighbors,legal_neighbors)
		next_index = np.random.choice(neighbors,1)[0]
		next_state = np.mean(self.net_brain_adj, axis=1)
		next_state = next_state[next_index]
		return next_state, next_index
	def step(self, action, index):
		self.model.train()
		self.optimizer.zero_grad()
		self.train(action, index)
		next_state, next_index = self.transition(action, index)
		val_acc = self.eval()
		benchmark = np.mean(np.array(self.past_performance[-self.benchmark_num:]))
		self.past_performance.append(val_acc)
		reward = val_acc - benchmark
		return next_state, next_index, reward, val_acc
	def train(self, act, index):
		self.model.train()
		pred = self.model((act, index))
		label = np.array([self.net_label[index]])
		label = torch.LongTensor(label).to(self.device)
		F.nll_loss(pred, label).backward()
		self.optimizer.step()
	def eval(self):
		self.model.eval()
		batch_dict = {}
		val_indexes = self.val_idx
		val_states = np.mean(self.net_brain_adj, axis=1)
		val_states = val_states[val_indexes]
		val_actions = self.policy.eval_step(val_states)
		for act, idx in zip(val_actions, val_indexes):
			if act not in batch_dict.keys():
				batch_dict[act] = []
			batch_dict[act].append(idx)
		val_acc = 0.
		for act in batch_dict.keys():
			indexes = batch_dict[act]
			if len(indexes) > 0:
				preds = self.model((act, indexes))
				preds = preds.max(1)[1]
				labels = torch.LongTensor(self.net_label[indexes]).to(self.device)
				val_acc += preds.eq(labels).sum().item()
		return val_acc/len(val_indexes)
