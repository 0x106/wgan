import torch, os, sys, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

class Map(dict):
	"""
	Example:
	m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
	Taken from: http://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
	- user: epool
	"""
	def __init__(self, *args, **kwargs):
		super(Map, self).__init__(*args, **kwargs)
		for arg in args:
			if isinstance(arg, dict):
				for k, v in arg.iteritems():
					self[k] = v

		if kwargs:
			for k, v in kwargs.iteritems():
				self[k] = v

	def __getattr__(self, attr):
		return self.get(attr)

	def __setattr__(self, key, value):
		self.__setitem__(key, value)

	def __setitem__(self, key, value):
		super(Map, self).__setitem__(key, value)
		self.__dict__.update({key: value})

	def __delattr__(self, item):
		self.__delitem__(item)

	def __delitem__(self, key):
		super(Map, self).__delitem__(key)
		del self.__dict__[key]

# these hold all the parameters and variables
opt, var, net = Map({}), Map({}), Map({})

opt.N = 100000
opt.B = 1000
opt.K = 1
opt.Z = 100
opt.ngf = 64
opt.ndf = 64
opt.epochs = 100
opt.optim = 'adam' # 'rmsprop'
opt.clamp_lower, opt.clamp_upper = -0.01, 0.01
opt.lr = 0.0005
opt.Diters = 100

assert( (opt.N % opt.B == 0) )

index = 1
while os.path.exists('./vault/'+str(index).zfill(2)):
	index += 1
os.makedirs('./vault/'+str(index).zfill(2))
opt.output_path = './vault/'+str(index).zfill(2)

os.system("cp main.py " + opt.output_path)

def data(_N):
	data = torch.FloatTensor(_N,opt.K)

	if opt.K == 1:
		data[_N//2:].normal_(0,1)
		data[:_N//2].normal_(10,1)
	if opt.K == 2: # 8 Gaussians
		pass

	targets = (torch.FloatTensor(_N)).fill_(0.)

	dataset = data_utils.TensorDataset(data, targets)
	dataloader = data_utils.DataLoader(dataset, batch_size=opt.B, shuffle=True)

	return dataloader

def weight_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
	    m.weight.data.normal_(0.0, 0.4)

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.main = nn.Sequential(
			nn.Linear(opt.Z, opt.ngf),
			nn.ReLU(True),
			nn.Linear(opt.ngf, opt.ngf),
			nn.ReLU(True),
			nn.Linear(opt.ngf, opt.ngf),
			nn.ReLU(True),
			nn.Linear(opt.ngf, opt.K),
		)

	def forward(self, x):
		return self.main(x)

class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()

		self.main = nn.Sequential(
			nn.Linear(opt.K, opt.ndf),
			nn.ReLU(True),
			nn.Linear(opt.ndf, opt.ndf),
			nn.ReLU(True),
			nn.Linear(opt.ndf, opt.ndf),
			nn.ReLU(True),
			nn.Linear(opt.ndf, 1),
		)

	def forward(self, x):
		return self.main(x).mean(0).view(1)

def test(generator, data, noise, index):
	generator.eval()
	output = generator(Variable(noise)).data.numpy()

	data = data.numpy()

	if opt.K == 1:
		plt.hist(data, bins=100, alpha=0.5)
		plt.hist(output, bins=100, alpha=0.5)

	if opt.K == 2:
		plt.scatter(data[:,0], data[:,1], marker='+')
		plt.scatter(output[:,0], output[:,1], marker='+')

	plt.savefig(opt.output_path+'/img-' + str(index) + '.png')
	plt.clf()

	generator.train()

def train(opt, var, dataloader, net):
	gen_iters = 0

	for epoch in range(opt.epochs):
		data_iter, i = iter(dataloader), 0

		while i < len(dataloader):
			for param in net.critic.parameters(): # reset requires_grad
				param.requires_grad = True # they are set to False below in netG update

			Diters, j = opt.Diters, 0

			if gen_iters < 25 or gen_iters % 50 == 0:
				Diters = 100

			while j < Diters and i < len(dataloader):
				i, j = i + 1, j + 1

				for p in net.critic.parameters():
					p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
				net.critic.zero_grad()

				data, _ = data_iter.next()

				# train real
				inputv = Variable(var.data.copy_(data))
				error_real = net.critic(inputv)
				error_real.backward(var.one)

				# train fake
				noisev = Variable(var.noise.normal_(0,1), volatile=True)
				output = Variable(net.generator(noisev).data)
				inputv = output
				error_fake = net.critic(inputv)
				error_fake.backward(var.mone)

				error = error_real - error_fake
				net.optD.step()

			for param in net.critic.parameters():
				param.requires_grad = False # to avoid computation

			# train generator
			net.generator.zero_grad()
			noisev = Variable(var.noise.normal_(0,1))

			output = net.generator(noisev)
			error_g = net.critic(output)
			error_g.backward(var.one)

			net.optG.step()

			gen_iters += 1

		# epoch
		test(net.generator, var.data, var.noise, epoch)
		print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
			% (epoch, opt.epochs, i, len(dataloader), gen_iters,
			error.data[0], error_g.data[0], error_real.data[0], error_fake.data[0]))

		var.logs.append([error.data[0], error_g.data[0], error_real.data[0], error_fake.data[0]])

# data
dataloader = data(opt.N)

# networks
net.generator = Generator()
net.critic = Critic()

net.generator.apply(weight_init)
net.critic.apply(weight_init)

# optimisers
if opt.optim == 'adam':
	net.optD = optim.Adam(net.critic.parameters(), lr=opt.lr)
	net.optG = optim.Adam(net.generator.parameters(), lr=opt.lr)
else:
	net.optD = optim.RMSprop(net.critic.parameters(), lr=opt.lr)
	net.optG = optim.RMSprop(net.generator.parameters(), lr=opt.lr)

# variables
var.data        = torch.FloatTensor(opt.B, opt.K)
var.noise       = torch.FloatTensor(opt.B, opt.Z)
var.fixed_noise = torch.FloatTensor(opt.B, opt.Z).normal_(0,1)

var.one  = torch.FloatTensor([1])
var.mone = var.one * -1

var.logs = []

print opt
# print var
print net

## =================================================== ##

train(opt, var, dataloader, net)

logs = np.zeros((len(var.logs), 4))
for i in range(len(var.logs)):
	for k in range(4):
		logs[i,k] = var.logs[i][k]

plt.plot(logs[:,0], label='loss')
plt.plot(logs[:,1], label='G')
plt.plot(logs[:,2], label='real')
plt.plot(logs[:,3], label='fake')
plt.legend()
plt.savefig(opt.output_path+'/logs.png')
plt.clf()

print("Training complete.")





#
