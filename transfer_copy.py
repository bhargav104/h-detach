import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm_cell import LSTM
from generator import generate_copying_sequence
from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='Transfer copying task')
parser.add_argument('--T', type=int, default=100, help='size of H truncations')
parser.add_argument('--model-dir', type=str, default='', help='path to model file')
args = parser.parse_args()	

#writer = SummaryWriter()

torch.manual_seed(555)
np.random.seed(555)
torch.cuda.manual_seed_all(555)
tensor = torch.FloatTensor

T = args.T
inp_size = 1
out_size = 9
test_size = 5000

def create_dataset(size, T):
	d_x = []
	d_y = []
	for i in range(size):
		sq_x, sq_y = generate_copying_sequence(T)
		sq_x, sq_y = sq_x[0], sq_y[0]
		d_x.append(sq_x)
		d_y.append(sq_y)

	d_x = torch.stack(d_x)
	d_y = torch.stack(d_y)
	return d_x, d_y


class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.lstm = LSTM(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state	

def test_model(model, test_x, test_y, criterion, hid_size):
	loss = 0
	accuracy = 0
	inp_x = torch.transpose(test_x, 0, 1)
	inp_y = torch.transpose(test_y, 0, 1)
	h = torch.zeros(test_size, hid_size).to(device)
	c = torch.zeros(test_size, hid_size).to(device)

	with torch.no_grad():
		for i in range(T + 20):
			output, (h, c) = model(inp_x[i], (h, c))
			loss += criterion(output, inp_y[i].squeeze(1)).item()
			if i >= T + 10:
				preds = torch.argmax(output, dim=1)
				actual = inp_y[i].squeeze(1)
				correct = preds == actual
				accuracy += correct.sum().item()

	loss /= (T + 20.0)
	accuracy /= (500.0)

	print('test loss {}, test accuracy {}'.format(loss,accuracy))
	return loss, accuracy


device = torch.device('cuda')
with open(args.model_dir + '/best_model.pt', 'rb') as f:
    state = torch.load(f)
net = state['net']
hid_size = net['hid_size']

criterion = nn.CrossEntropyLoss()
test_x, test_y = create_dataset(test_size, T)	
test_x, test_y = test_x.to(device), test_y.to(device)

test_model(net, test_x, test_y, criterion, hid_size)	
#writer.close()