import torch
import math
import sys
import numpy as np
import torch.nn as nn
from lstm_cell import LSTM
import torch.nn.functional as F
import torch.optim as optim
from generator import generate_copying_sequence
from tensorboardX import SummaryWriter
import argparse
import os
import glob
import tqdm 
parser = argparse.ArgumentParser(description='Copying Task')
parser.add_argument('--p-detach', type=float, default=-1.0, help='probability of detaching each timestep')
parser.add_argument('--lstm-size', type=int, default=128, help='hidden size of LSTM')
parser.add_argument('--save-dir', type=str, default='default', help='save dir of the results')
parser.add_argument('--seed', type=int, default=3, help='seed value')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping norm')
parser.add_argument('--T', type=int, default=300, help='T')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

args = parser.parse_args()
log_dir = '/directory/to/save/experiments/'+args.save_dir + '/'


if os.path.isdir(log_dir):
	if len(glob.glob(log_dir+'events.*'))>0:
		print ('TensorBoard file exists by this name. Please delete it manually using \nrm -f {} \nor choose another save_dir.'.format(glob.glob(log_dir+'events.*')[0]))
		exit(0)

writer = SummaryWriter(log_dir=log_dir)

device = torch.device('cuda')
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
tensor = torch.FloatTensor

n_epochs = args.n_epochs
T = args.T
batch_size = args.batch_size
hid_size = args.lstm_size
lr = args.lr

inp_size = 1
out_size = 9
train_size = 100000
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

def test_model(model, test_x, test_y, criterion):
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

	print('validation loss {}, validation accuracy {}'.format(loss,accuracy))
	return loss, accuracy

def train_model(model, epochs, criterion, optimizer):

	train_x, train_y = create_dataset(train_size, T)
	test_x, test_y = create_dataset(test_size, T)
	train_x, train_y = train_x.to(device), train_y.to(device)
	test_x, test_y = test_x.to(device), test_y.to(device)
	global best_acc, ctr, start_epoch

	iters = -1
	p_detach=0.
	for epoch in range(start_epoch, epochs):
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0
		for z in tqdm.tqdm(range(train_size // batch_size), total=train_size // batch_size):
			iters += 1
			ind = np.random.choice(train_size, batch_size)
			inp_x, inp_y = train_x[ind], train_y[ind]
			inp_x.transpose_(0, 1)
			inp_y.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)

			sq_len = T + 20
			loss = 0
			val = np.random.random(size=1)[0]
			for i in range(sq_len):
				if args.p_detach>0:
					p_detach = args.p_detach	
					rand_val = np.random.random(size=1)[0]
					if rand_val <= p_detach:
						h = h.detach()
				output, (h, c) = model(inp_x[i], (h, c))
				loss += criterion(output, inp_y[i].squeeze(1))

			loss /= (1.0 * sq_len)
			model.zero_grad()
			loss.backward()
			norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip>0 else float('inf'))
			
			optimizer.step()

			loss_val = loss.item()
			writer.add_scalar('/hdetach:train_loss', loss_val, ctr)
			ctr += 1



		t_loss, accuracy = test_model(model, test_x, test_y, criterion)
		if accuracy > best_acc:
			best_acc = accuracy
			print('best validation accuracy ' + str(best_acc))
		writer.add_scalar('/hdetach:val_acc', accuracy, epoch)

		# print('Saving per epoch model..')
		# state = {
  #       'net': net,
  #       'best_acc': best_acc,
  #       'ctr': ctr,
  #       'start_epoch': epoch,
  #   	}
		# with open(log_dir + '/last_model.pt', 'wb') as f:
		# 	torch.save(state, f)


print('==> Building model..')
net = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.CrossEntropyLoss()

start_epoch=0
ctr=0
best_acc=0


optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
#writer.close()
