import torch
import argparse
from model.gnn import gnn_env
from model.dqn import QAgent
from copy import deepcopy
import numpy as np


parser = argparse.ArgumentParser(description='BN-GNN')
parser.add_argument('--dataset', type=str, default="BP")
parser.add_argument('--view', type=str, default="fmri")
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--knn', type=int, default=6)
parser.add_argument('--action_num', type=int, default=3)
parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--slope', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--gnn_type', type=str, default='GCN')
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--max_timesteps', type=int, default=10)
parser.add_argument('--max_episodes0', type=int, default=1)
parser.add_argument('--max_episodes1', type=int, default=200)
parser.add_argument('--discount_factor', type=float, default=0.95)
parser.add_argument('--epsilon_start', type=float, default=1.0)
parser.add_argument('--epsilon_end', type=float, default=0.1)
parser.add_argument('--epsilon_decay_steps', type=int, default=10)
parser.add_argument('--benchmark_num', type=int, default=20)
parser.add_argument('--replay_memory_size', type=int, default=10000)
parser.add_argument('--replay_memory_init_size', type=int, default=500)
parser.add_argument('--update_target_estimator_every', type=int, default=2)
parser.add_argument('--norm_step', type=int, default=100)
parser.add_argument('--mlp_layers', type=list, default=[128, 64, 32])
args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
	acc_records = []
	for repeat in range(args.repeats):
		env = gnn_env(dataset = args.dataset,
					  view = args.view,
					  knn = args.knn,
					  folds = args.folds,
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
					   batch_size=env.batch_size_qdn,
					   num_net = env.num_net,
					   action_num=env.action_num,
					   norm_step=args.norm_step,
					   mlp_layers=args.mlp_layers,
					   state_shape=env.state_shape,
					   device=args.device)
		env.policy = agent
		last_val = 0.0
		print("Training Meta-policy on Validation Set")
		for i_episode in range(args.max_episodes0):
			loss, _, (val_acc, mean_reward) = agent.learn(env, args.max_timesteps)
			if val_acc >= last_val:
				best_policy = deepcopy(agent)
			last_val = val_acc
			# print("Training Meta-policy:", i_episode, "Val_ACC:", val_acc, "mean_reward:", mean_reward)


		env.policy = best_policy
		states = env.reset()
		max_test_acc = 0.0
		max_val_acc = 0.0
		for i_episode in range(1, args.max_episodes1):
			actions = env.policy.eval_step(states)
			states, rewards, dones, (val_acc, mean_reward) = env.step(actions)
			test_acc = env.test()
			if val_acc > max_val_acc:
				max_val_acc = val_acc
				max_test_acc = test_acc
			elif val_acc == max_val_acc and test_acc > max_test_acc:
				max_test_acc = test_acc
			# print("Epoch", i_episode, "; val_acc:", val_acc, "; test_acc:", test_acc)

		acc_records.append(max_test_acc)
		print("Training GNN", repeat, "; Val_Acc:", max_val_acc, "; Test_Acc:", max_test_acc)
	mean_acc = np.mean(np.array(acc_records))
	std_acc = np.std(np.array(acc_records))
	print("Mean Acc:", mean_acc,'±',std_acc)

if __name__ == '__main__':
	main()