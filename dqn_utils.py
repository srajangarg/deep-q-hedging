import torch
from machin.frame.algorithms import DQN
from machin.model.nets import static_module_wrapper

def create_dqn(num_hidden, hidden_dim, discount, epsilon_decay, learning_rate, env):
	# TODO add batch norm?
	class QNet(torch.nn.Module):
	    def __init__(self, state_dim, num_hidden, hidden_dim, action_dim):
	        super(QNet, self).__init__()
	        self.input  = torch.nn.Linear(state_dim, hidden_dim)
	        self.hidden = [torch.nn.Linear(hidden_dim, hidden_dim)] * num_hidden
	        self.output = torch.nn.Linear(hidden_dim, action_dim)

	    def forward(self, state):
	        res = torch.relu(self.input(state))
	        for hidden_layer in self.hidden:
	            res = torch.relu(hidden_layer(res))
	        return self.output(res)

	qnet   = QNet(env.state_size(), num_hidden, hidden_dim, env.OCS+1)
	qnet_t = QNet(env.state_size(), num_hidden, hidden_dim, env.OCS+1)
	qnet   = static_module_wrapper(qnet, "cpu", "cpu")
	qnet_t = static_module_wrapper(qnet_t, "cpu", "cpu")

	# TODO add LR scheduler?
	return DQN(qnet, qnet_t, torch.optim.Adam, torch.nn.MSELoss(reduction='sum'),
	           discount=discount, epsilon_decay=epsilon_decay, learning_rate=learning_rate,
	           lr_scheduler=torch.optim.lr_scheduler.StepLR, lr_scheduler_kwargs=[{"step_size": 500*128, "gamma" : 0.5}])

def predict_while_testing(dqn, state):
	out = dqn.act_discrete({"state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)})
	return out

def predict_while_training(dqn, state):
	out = dqn.act_discrete_with_noise({"state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)})
	return out

def add_to_training_set(dqn, state, action, new_state, reward, done):
	dqn.store_episode([{
	    "state": {"state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)},
	    "action": {"action": action},
	    "next_state": {"state": torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)},
	    "reward":  reward,
	    "terminal": done
	}])

def train(dqn):
	dqn.update()