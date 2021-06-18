import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from random import shuffle
from collections import defaultdict
import scipy.io as sio


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
		g_num, n_num, n_num = adjs.shape
		adjs = adjs.detach()
		for i in range(g_num):
			adjs[i] += torch.eye(n_num)
			adjs[i][adjs[i]>0.] = 1.
			degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)
			degree_matrix = torch.pow(degree_matrix,-1)
			degree_matrix[degree_matrix == float("inf")] = 0.
			degree_matrix = torch.diag(degree_matrix)
			adjs[i] = torch.mm(degree_matrix, adjs[i])
		return adjs

	def forward(self, input):
		action, batch_idx = input
		batch_features = self.features[batch_idx]
		adjs = self.adjs[batch_idx]
		batch_features = self.fc1(torch.matmul(adjs, batch_features))
		batch_features = self.leaky_relu(batch_features)
		batch_features = self.dropout(batch_features)
		if action == '1':
			batch_features = self.fc2(torch.matmul(adjs, batch_features))
			batch_features = self.leaky_relu(batch_features)
			batch_features = self.dropout(batch_features)
		elif action == '2':
			batch_features = self.fc2(torch.matmul(adjs, batch_features))
			batch_features = self.leaky_relu(batch_features)
			batch_features = self.dropout(batch_features)
			batch_features = self.fc3(torch.matmul(adjs, batch_features))
			batch_features = self.leaky_relu(batch_features)
			batch_features = self.dropout(batch_features)
		predicts = torch.mean(batch_features,dim=1).squeeze(1)
		predicts = self.classifier(predicts)
		predicts = F.log_softmax(predicts, dim=1)
		batch_features = torch.mean(batch_features,dim=1).squeeze()
		return batch_features, predicts

class gnn_env(object):
	def __init__(self,
				 dataset,
				 view,
				 knn,
				 folds,
				 max_layer,
				 hid_dim,
				 out_dim,
				 drop,
				 slope,
				 lr,
				 weight_decay,
				 gnn_type,
				 device,
				 policy,
				 benchmark_num):
		self.dataset = dataset
		self.view = view
		self.knn = knn
		self.folds = folds
		self.max_layer = max_layer
		self.action_num = max_layer
		self.device = device
		self.policy = policy
		self.benchmark_num = benchmark_num
		self.load_dataset()
		self.num_train, self.num_val, self.num_test \
			= len(self.train_idx), len(self.val_idx), len(self.test_idx)
		self.num_net = self.num_train + self.num_val + self.num_test
		if gnn_type == 'GCN':
			self.model = GCN(self.init_net_feats, self.reli_net_adjs, hid_dim, out_dim, drop, slope).to(device)
		# elif gnn_type == 'GAT':
			# self.model = GAT(self.init_net_feats, self.reli_net_adjs, out_dim, slope, 0.3, [1, 1, 1, 1], [hid_dim] * max_layer, max_layer)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
		self.batch_size_qdn = self.num_train
		self.state_shape = self.reset().shape
		self.gnn_buffers = defaultdict(list)
		self.past_performance = [0]

	def load_dataset(self):
		if self.dataset == 'BP' or self.dataset == 'HIV':
			self.BP_HIV_00()
		elif self.dataset == 'ADHD' or self.dataset == 'HI' or self.dataset == 'GD':
			self.ADHD_HI_GD_00()
		elif self.dataset == 'EEG':
			self.EEG_00()

	def EEG_00(self):
		raw_data = sio.loadmat('./data/raw/' + self.dataset + '/' + self.dataset + '.mat')
		tmp_net_adjs = raw_data['corr_matrix']
		init_net_adjs_0 = tmp_net_adjs[:, :, 0, :21].squeeze().transpose(2, 0, 1)
		init_net_adjs_1 = tmp_net_adjs[:, :, 1, 21:].squeeze().transpose(2, 0, 1)
		reli_net_adjs = np.vstack((init_net_adjs_0, init_net_adjs_1))
		net_labels = [0] * 21
		net_labels.extend([1] * 40)
		self.net_labels = np.array(net_labels)
		num_net = len(net_labels)
		num_val = math.ceil(num_net / self.folds)
		num_train = num_net - num_val * 2
		all_idx = [idx for idx in range(num_net)]
		shuffle(all_idx)
		self.train_idx = all_idx[:num_train]
		self.val_idx = all_idx[num_train:num_train + num_val]
		self.test_idx = all_idx[num_train + num_val:]
		self.init_net_feats = np.random.random(reli_net_adjs.shape)
		self.reli_net_adjs = reli_net_adjs

	def ADHD_HI_GD_00(self):
		fp_ori = open('./data/raw/' + self.dataset + '/' + self.dataset + '.nel', "r")
		fp = fp_ori.read().split('\n')
		fp_ori.close()
		e_net = [[]]
		idx_net = 0
		net_labels = []
		n_new2old_net = [{}]
		for line in fp:
			if line.strip() == '':
				e_net.append([])
				n_new2old_net.append({})
				idx_net += 1
			else:
				entries = line.split(' ')
				if entries[0] == 'x':
					label = int(entries[1]) if int(entries[1]) > 0 else 0
					net_labels.append(label)
				elif entries[0] == 'n':
					n_new_id = entries[1]
					n_old_id = entries[2].split('_')[-1]
					n_new2old_net[idx_net][n_new_id] = n_old_id
				elif entries[0] == 'e':
					e_net[idx_net].append(entries)
		e_net = e_net[:-2]
		n_new2old_net = n_new2old_net[:-2]
		self.net_labels = np.array(net_labels)
		reli_net_adjs = np.zeros((len(e_net), 200, 200))
		for i in range(len(e_net)):
			edges = e_net[i]
			new2old_dict = n_new2old_net[i]
			for edge in edges:
				x = int(new2old_dict[edge[1]])
				y = int(new2old_dict[edge[2]])
				reli_net_adjs[i][x][y] = 1
				reli_net_adjs[i][y][x] = 1
		num_net = len(net_labels)
		num_val = math.ceil(num_net / self.folds)
		num_train = num_net - num_val * 2
		all_idx = [idx for idx in range(num_net)]
		shuffle(all_idx)
		self.train_idx = all_idx[:num_train]
		self.val_idx = all_idx[num_train:num_train + num_val]
		self.test_idx = all_idx[num_train + num_val:]
		self.init_net_feats = np.random.random(reli_net_adjs.shape)
		self.reli_net_adjs = reli_net_adjs


	def BP_HIV_00(self):
		raw_data = sio.loadmat('./data/raw/' + self.dataset + '/' + self.dataset + '.mat')
		net_labels = raw_data['label'].squeeze()
		net_labels[net_labels < 0] = 0
		self.net_labels = net_labels
		raw_fmri = np.expand_dims(raw_data['dti'].transpose(2, 0, 1), 1)
		raw_dti = np.expand_dims(raw_data['fmri'].transpose(2, 0, 1), 1)
		self.init_net_adjs = np.concatenate((raw_dti, raw_fmri), axis=1)
		self.init_net_feats = self.BP_HIV_01()
		self.reli_net_adjs = self.BP_HIV_02()
		if self.view == 'dti':
			self.init_net_feats = raw_dti[:, 0, :, :]
			self.reli_net_adjs = self.init_net_adjs[:, 0, :, :]
		elif self.view == 'fmri':
			self.init_net_feats = raw_fmri[:, 0, :, :]
			self.reli_net_adjs = self.init_net_adjs[:, 1, :, :]
		self.train_idx, self.val_idx, self.test_idx = self.BP_HIV_03()

	def BP_HIV_01(self):
		N, V, R, F = self.init_net_adjs.shape
		sum_X = 0
		for i in range(N):
			for j in range(V):
				x = self.init_net_adjs[i, j, :, :]
				sum_X = sum_X + np.matmul(x, x.transpose())
		u, _, _ = np.linalg.svd(sum_X)
		return u

	def BP_HIV_02(self):
		S, _, N, N = self.init_net_adjs.shape
		reli_net_adjs = np.zeros((S, N, N))
		for s in range(S):
			if self.view == 'dti':
				imp_matrix = np.abs(self.init_net_adjs[s, 0, :, :])
			elif self.view == 'fmri':
				imp_matrix = np.abs(self.init_net_adjs[s, 1, :, :])
			adj_matrix = np.zeros((N, N))
			tmp_matrix = np.zeros((N, N))
			for i in range(N):
				tmp_matrix[i][np.argpartition(imp_matrix[i], -self.knn)[-self.knn:]] = 1
				tmp_matrix[i][tmp_matrix[i] != 1] = 0
			tmp_matrix = tmp_matrix + np.eye(N)
			tmp_matrix[tmp_matrix > 1] = 1
			for i in range(N):
				for j in range(N):
					if tmp_matrix[i][j] == 1:
						adj_matrix[i][j] = np.exp(-(1 / 2) * np.linalg.norm(self.init_net_feats[j] - self.init_net_feats[i], ord=2))
				reli_net_adjs[s, i] = adj_matrix[i]
		return reli_net_adjs


	def BP_HIV_03(self):
		self.num_net = len(self.net_labels)
		num_val = math.ceil(self.num_net / self.folds)
		num_train = self.num_net - num_val*2
		all_idx = [idx for idx in range(self.num_net)]
		shuffle(all_idx)
		train_idx = all_idx[:num_train]
		val_idx = all_idx[num_train:num_train + num_val]
		test_idx = all_idx[num_train + num_val:]
		return train_idx, val_idx, test_idx


	def reset(self):
		states = np.mean(self.init_net_feats,axis=1)
		states = states[self.train_idx]
		self.optimizer.zero_grad()
		return states


	def step(self, actions):
		self.model.train()
		self.optimizer.zero_grad()
		index = self.train_idx
		done = False
		for act, idx in zip(actions, index):
			self.gnn_buffers[act].append(idx)
			if len(self.gnn_buffers[act]) >= self.batch_size_qdn:
				self.train(act, self.gnn_buffers[act])
				self.gnn_buffers[act] = []
				done = True

		next_states = self.reset()
		val_acc_dict = self.eval()
		val_acc = [val_acc_dict[a] for a in actions]
		benchmark = np.mean(np.array(self.past_performance[-self.benchmark_num:]))
		self.past_performance.extend(val_acc)
		reward = [each - benchmark for each in val_acc]
		r = np.mean(np.array(reward))
		val_acc = np.mean(val_acc)
		return next_states, reward, [done] * len(next_states), (val_acc, r)


	def train(self, act, indexes):
		self.model.train()
		_, preds = self.model((act, indexes))
		labels = torch.LongTensor(self.net_labels[indexes])
		F.nll_loss(preds, labels).backward()
		self.optimizer.step()

	def eval(self):
		self.model.eval()
		batch_dict = {}
		index = self.val_idx
		val_states = np.mean(self.init_net_feats, axis=1)
		val_states = val_states[index]
		val_acts = self.policy.eval_step(val_states)
		for i, a in zip(index, val_acts):
			if a not in batch_dict.keys():
				batch_dict[a] = []
			batch_dict[a].append(i)

		acc = {a: 0.0 for a in range(self.max_layer)}
		for act in batch_dict.keys():
			indexes = batch_dict[act]
			if len(indexes) > 0:
				features,logits = self.model((act, indexes))
				preds = logits.max(1)[1]
				labels = torch.LongTensor(self.net_labels[indexes])
				acc[act] = preds.eq(labels).sum().item() / len(indexes)
		return acc

	def test(self):
		self.model.eval()
		batch_dict = {}
		index = self.test_idx
		test_states = np.mean(self.init_net_feats, axis=1)
		test_states = test_states[index]
		test_acts = self.policy.eval_step(test_states)
		for i, a in zip(index, test_acts):
			if a not in batch_dict.keys():
				batch_dict[a] = []
			batch_dict[a].append(i)

		acc = 0.0
		acc_cnt = 0
		for act in batch_dict.keys():
			indexes = batch_dict[act]
			if len(indexes) > 0:
				features, logits = self.model((act, indexes))
				preds = logits.max(1)[1]
				labels = torch.LongTensor(self.net_labels[indexes])
				acc += preds.eq(labels).sum().item() / len(indexes)
				acc_cnt += 1
		acc = acc/acc_cnt
		return acc








