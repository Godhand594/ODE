# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:47:36 2021

@author: User
"""

import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np 

C3S = {'K1':1.5,
       'N1':0.7,
       'K2':0.05,
       'K3':1.1,
       'N3':3.3,
       'H':1.8,
       'Ea':41570}

C2S = {'K1':0.5,
       'N1':1.0,
       'K2':0.02,
       'K3':0.7,
       'N3':5.0,
       'H':1.35,
       'Ea':20785}

C3A = {'K1':1.0,
       'N1':0.85,
       'K2':0.04,
       'K3':1.0,
       'N3':3.2,
       'H':1.60,
       'Ea':54040}

C4AF = {'K1':0.37,
       'N1':0.7,
       'K2':0.015,
       'K3':0.4,
       'N3':3.7,
       'H':1.45,
       'Ea':34087}

'''其餘參數'''
surface_area = 300.0
Reference_area = 385.0
Reference_T = 293.15
R=8.314
rh=1.0
dt=0.1
T=293.15


# Define the NN model to solve the problem
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), 
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
            nn.Sigmoid())
 
    def forward(self, x):
        return self.net(x)
 
model = Model()
 
# Define loss_function from the Ordinary differential equation to solve
def ODE(x,y):
    dydx, = torch.autograd.grad(y, x, 
    grad_outputs=y.data.new(y.shape).fill_(1),
    create_graph=True, retain_graph=True)
 
    eq = dydx-C3S['K2']*(1-y)**(2/3)/(1-(1-y)**(1/3)) \
        *((rh-0.55)/0.45)**(4) \
        *torch.exp(torch.tensor(C3S['Ea']*(1/Reference_T-1/T)/R))
    ic = model(torch.tensor([0.])) - 0.00001    # y(x=0) = 0.00001
    return torch.mean(eq**2) + ic**2
 
loss_func = ODE
 
# Define the optimization
# opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.99,nesterov=True) # Equivalent to blog
opt = optim.Adam(model.parameters(),lr=0.0001,amsgrad=True) # Got faster convergence with Adam using amsgrad
 
# Define reference grid 
x_data = torch.linspace(0,365,3650,requires_grad=True)
x_data = x_data.view(3650,1) # reshaping the tensor
 
# Iterative learning
epochs = 1000
for epoch in range(epochs):
    opt.zero_grad()
    y_trial = model(x_data)
    loss = loss_func(x_data, y_trial)
 
    loss.backward()
    opt.step()
 
    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
Y_val=y_trial.data.numpy() 
# Plot Results
#plt.plot(x_data.data.numpy(), np.exp(-x_data.data.numpy()**2), label='exact')
plt.plot(x_data.data.numpy(), y_trial.data.numpy(), label='approx')
plt.xscale('log')#X軸轉乘對數
plt.legend()
plt.show()