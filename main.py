import torch
import argparse
from model.gnn import gnn_env, GCN
from model.dqn import QAgent
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve,auc

parser = argparse.ArgumentParser(description='BN-GNN')
parser.add_argument('--dataset', type=str, default="BP")
parser.add_argument('--view', type=str, default="DTI")
parser.add_argument('--action_num', type=int, default=3)
parser.add_argument('--hid_dim', type=int, default=512)
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--slope', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--gnn_type', type=str, default='GCN')
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--max_timesteps', type=int, default=1000)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--discount_factor', type=float, default=0.95)
parser.add_argument('--epsilon_start', type=float, default=1.0)
parser.add_argument('--epsilon_end', type=float, default=0.05)
parser.add_argument('--epsilon_decay_steps', type=int, default=20)
parser.add_argument('--benchmark_num', type=int, default=20)
parser.add_argument('--replay_memory_size', type=int, default=10000)
parser.add_argument('--replay_memory_init_size', type=int, default=500)
parser.add_argument('--memory_batch_size', type=int, default=20)
parser.add_argument('--update_target_estimator_every', type=int, default=1)
parser.add_argument('--norm_step', type=int, default=100)
parser.add_argument('--mlp_layers', type=list, default=[128, 64, 32])
args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
	acc_records = []
	auc_records = []
	for repeat in range(args.repeats):
		env = gnn_env(dataset = args.dataset,
					  view = args.view,
					  max_layer = args.action_num,
					  hid_dim = args.hid_dim,
					  out_dim = args.out_dim,
					  drop = args.dropout,
					  slope = args.slope,
					  lr = args.lr,
					  weight_decay = args.weight_decay,
					  gnn_type = args.gnn_type,
					  device = args.device,
					  policy = "",
					  benchmark_num = args.benchmark_num)
		agent = QAgent(replay_memory_size = args.replay_memory_size,
					   replay_memory_init_size = args.replay_memory_init_size,
					   update_target_estimator_every = args.update_target_estimator_every,
					   discount_factor = args.discount_factor,
					   epsilon_start = args.epsilon_start,
					   epsilon_end = args.epsilon_end,
					   epsilon_decay_steps = args.epsilon_decay_steps,
					   lr=args.lr,
					   batch_size=args.memory_batch_size,
					   num_net = env.num_net,
					   action_num=env.action_num,
					   norm_step=args.norm_step,
					   mlp_layers=args.mlp_layers,
					   state_shape=env.state_shape,
					   device=args.device)
		env.policy = agent
		agent.learn(env, args.max_timesteps)
		GNN2 = GCN(env.init_net_feat, env.net_brain_adj, args.hid_dim, args.out_dim, args.dropout, args.slope).to(args.device)
		# GNN2 = GCN(env.init_net_feat, env.init_net_feat, args.hid_dim, args.out_dim, args.dropout, args.slope).to(args.device)
		GNN_lr = 0.005
		if args.dataset == 'HIV' and args.view == 'DTI':
			GNN_wd = 0.05
		elif args.dataset == 'HIV' and args.view == 'fMRI':
			GNN_wd = 0.001
		elif args.dataset == 'BP' and args.view == 'DTI':
			GNN_wd = 0.05
		elif args.dataset == 'BP' and args.view == 'fMRI':
			GNN_wd = 0.05
		elif args.dataset == 'ADHD':
			GNN_wd = 0.001
		elif args.dataset == 'HI':
			GNN_wd = 0.05
		elif args.dataset == 'GD':
			GNN_wd = 0.001
		elif args.dataset == 'HA':
			GNN_wd = 0.001
		Optimizer = torch.optim.Adam(GNN2.parameters(), lr=GNN_lr, weight_decay=GNN_wd)
		train_gnn_buffer = defaultdict(list)
		val_gnn_buffer = defaultdict(list)
		test_gnn_buffer = defaultdict(list)
		states = np.mean(env.net_brain_adj, axis=1)
		train_states = states[env.train_idx]
		val_states = states[env.val_idx]
		test_states = states[env.test_idx]
		train_actions = env.best_policy.eval_step(train_states)
		val_actions = env.best_policy.eval_step(val_states)
		test_actions = env.best_policy.eval_step(test_states)
		for act, idx in zip(train_actions, env.train_idx):
			train_gnn_buffer[act].append(idx)
		for act, idx in zip(val_actions, env.val_idx):
			val_gnn_buffer[act].append(idx)
		for act, idx in zip(test_actions, env.test_idx):
			test_gnn_buffer[act].append(idx)
		max_val_acc = 0.
		max_test_acc =0.
		max_test_auc =0.
		for epoch in range(1, args.epoch_num):
			train_loss = 0.
			val_loss = 0.
			val_pred_list = []
			val_true_list = []
			test_pred_list = []
			test_true_list = []
			GNN2.train()
			for act in range(args.action_num):
				indexes = train_gnn_buffer[act]
				if len(indexes) > 0:
					preds = GNN2((act, indexes))
					labels = np.array(env.net_label[indexes])
					labels = torch.LongTensor(labels).to(args.device)
					loss = F.nll_loss(preds, labels)
					train_loss += loss
					loss.backward()
					Optimizer.step()
			GNN2.eval()
			for act in range(args.action_num):
				indexes = val_gnn_buffer[act]
				if len(indexes) > 0:
					preds = GNN2((act, indexes))
					labels = np.array(env.net_label[indexes])
					labels = torch.LongTensor(labels).to(args.device)
					loss = F.nll_loss(preds, labels)
					val_loss += loss
					preds = preds.max(1)[1]
					val_pred_list.extend(preds.to('cpu').numpy())
					val_true_list.extend(labels.to('cpu').numpy())
			for act in range(args.action_num):
				indexes = test_gnn_buffer[act]
				if len(indexes) > 0:
					preds = GNN2((act, indexes))
					labels = np.array(env.net_label[indexes])
					labels = torch.LongTensor(labels).to(args.device)
					preds = preds.max(1)[1]
					test_pred_list.extend(preds.to('cpu').numpy())
					test_true_list.extend(labels.to('cpu').numpy())
			val_pred_list = np.array(val_pred_list)
			val_true_list = np.array(val_true_list)
			val_acc = accuracy_score(val_true_list, val_pred_list)
			test_pred_list = np.array(test_pred_list)
			test_true_list = np.array(test_true_list)
			test_acc = accuracy_score(test_true_list, test_pred_list)
			test_fpr, test_tpr, _ = roc_curve(test_true_list, test_pred_list)
			test_auc = auc(test_fpr, test_tpr)
			if val_acc > max_val_acc:
				max_val_acc = val_acc
				max_test_acc = test_acc
				max_test_auc = test_auc
				# print("Epoch: {}".format(epoch), " Train_Acc: {:.5f}".format(train_acc), " Val_Acc: {:.5f}".format(max_val_acc), " Test_Acc: {:.5f}".format(max_test_acc))
			elif val_acc == max_val_acc:
				if test_acc > max_test_acc:
					max_test_acc = test_acc
					max_test_auc = test_auc
					# print("Epoch: {}".format(epoch), " Train_Acc: {:.5f}".format(train_acc), " Val_Acc: {:.5f}".format(max_val_acc), " Test_Acc: {:.5f}".format(max_test_acc))
		print("Test_Acc: {:.5f}".format(max_test_acc))
		print("Test_AUC: {:.5f}".format(max_test_auc))
		acc_records.append(max_test_acc)
		auc_records.append(max_test_auc)
		print('----------------------------------------------')
	mean_acc = np.mean(np.array(acc_records))
	std_acc = np.std(np.array(acc_records))
	print("Acc: {:.5f}".format(mean_acc),'± {:.5f}'.format(std_acc))
	mean_auc = np.mean(np.array(auc_records))
	std_auc = np.std(np.array(auc_records))
	print("AUC: {:.5f}".format(mean_auc),'± {:.5f}'.format(std_auc))

if __name__ == '__main__':
	for i in range(40):
		main()