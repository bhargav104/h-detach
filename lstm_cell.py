import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class LSTM(nn.Module):
	def __init__(self, inp_size, hidden_size):
		super().__init__()
		self.inp_size = inp_size
		self.hidden_size = hidden_size
		
		self.i2h = nn.Linear(inp_size, 4 * hidden_size)
		self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
		self.reset_parameters()


	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)

	def forward(self, x, hid_state):
		h, c = hid_state
		preact = self.i2h(x) + self.h2h(h)

		gates = preact[:, :3 * self.hidden_size].sigmoid()
		g_t = preact[:, 3 * self.hidden_size:].tanh()
		i_t = gates[:, :self.hidden_size]
		f_t = gates[:, self.hidden_size:2 * self.hidden_size]
		o_t = gates[:, -self.hidden_size:]

		c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
		h_t = torch.mul(o_t, c_t.tanh())

		return h_t, (h_t, c_t)