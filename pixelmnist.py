import torch
import math
import sys
import numpy as np
import torch.nn as nn
from lstm_cell import LSTM
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision as T
import argparse
import os
import glob
import tqdm

parser = argparse.ArgumentParser(description='sequential MNIST parameters')
parser.add_argument('--p-detach', type=float, default=-1.0, help='probability of detaching each timestep')
parser.add_argument('--permute', action='store_true', default=False, help='pMNIST or normal MNIST')
parser.add_argument('--save-dir', type=str, default='default', help='save directory')
parser.add_argument('--lstm-size', type=int, default=100, help='width of LSTM')
parser.add_argument('--seed', type=int, default=400, help='seed value')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for adam')
parser.add_argument('--clipval', type=float, default=1., help='gradient clipping value')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--anneal-p', type=int, default=0, help='number of epochs before total number of epochs for setting p-detach to 0')


args = parser.parse_args()
# log_dir = '/directory/to/save/experiments/'+args.save_dir + '/'


# if os.path.isdir(log_dir):
# 	if len(glob.glob(log_dir+'events.*'))>0:
# 		print ('TensorBoard file exists by this name. Please delete it manually using \nrm -f {} \nor choose another save_dir.'.format(glob.glob(log_dir+'events.*')[0]))
# 		exit(0)

writer = SummaryWriter(log_dir=log_dir)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
tensor = torch.FloatTensor

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000))
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000, 60000))

trainset = T.datasets.MNIST(root='./MNIST', train=True, download=True, transform=T.transforms.ToTensor())
testset = T.datasets.MNIST(root='./MNIST', train=False, download=True, transform=T.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, sampler=train_sampler, num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, sampler=valid_sampler, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2)

n_epochs = args.n_epochs
batch_size = args.batch_size
hid_size = args.lstm_size
lr = args.lr

T = 784
inp_size = 1
out_size = 10
train_size = 60000
test_size = 10000
clipval = float(args.clipval) if args.clipval>0 else float('inf')

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		self.lstm = LSTM(inp_size, hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x, new_state = self.lstm(x, state)
		x = self.fc1(x)
		return x, new_state	

def test_model(model, loader, criterion, order):
	
	accuracy = 0
	loss = 0
	with torch.no_grad():
		for i, data in enumerate(loader, 1):
			test_x, test_y = data
			test_x = test_x.view(-1, 784, 1)
			test_x, test_y = test_x.to(device), test_y.to(device)
			test_x.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)

			for j in order:
				output, (h, c) = model(test_x[j], (h, c))

			loss += criterion(output, test_y).item()
			preds = torch.argmax(output, dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()

	accuracy /= 100.0
	loss /= 100.0
	return loss, accuracy

def train_model(model, epochs, criterion, optimizer):

	best_acc = 0.0
	ctr = 0	
	global lr
	if args.permute:
		order = np.random.permutation(T)
	else:
		order = np.arange(T)

	test_acc = 0
	for epoch in range(epochs):
		if epoch>epochs-args.anneal_p:
			args.p_detach=-1
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0.
		iter_ctr = 0.
		for data in tqdm.tqdm(trainloader):
			iter_ctr+=1.
		# for z, data in enumerate(trainloader, 0):
			inp_x, inp_y = data
			inp_x = inp_x.view(-1, 28*28, 1)
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)
			inp_x.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)
			sq_len = T
			loss = 0

			for i in order:
				if args.p_detach >0:
					val = np.random.random(size=1)[0]
					if val <= args.p_detach:
						h = h.detach()
				output, (h, c) = model(inp_x[i], (h, c))

			loss += criterion(output, inp_y)

			model.zero_grad()
			loss.backward()
			norms = nn.utils.clip_grad_norm_(model.parameters(), clipval)

			optimizer.step()


			loss_val = loss.item()
			epoch_loss += loss_val
			# print(z, loss_val)
			# writer.add_scalar('/hdetach:loss', loss_val, ctr)
			ctr += 1

		v_loss, v_accuracy = test_model(model, validloader, criterion, order)
		if best_acc < v_accuracy:
			best_acc = v_accuracy
			_, test_acc = test_model(model, testloader, criterion, order)
		print('epoch_loss: {}, val accuracy: {} '.format(epoch_loss/(iter_ctr), v_accuracy))
		writer.add_scalar('/hdetach:val_acc', v_accuracy, epoch)
		writer.add_scalar('/hdetach:epoch_loss', epoch_loss/(iter_ctr), epoch)

	print('best val accuracy: {} '.format( best_acc))
	writer.add_scalar('/hdetach:best_val_acc', best_acc, 0)
	print('test accuracy: {} '.format( test_acc))
	writer.add_scalar('/hdetach:test_acc', test_acc, 0)

device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
writer.close()