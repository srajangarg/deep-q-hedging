import torch

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