
from __future__ import print_function

import numpy as np

class SGD ():
  def __init__(self, params, lr=0.001, reg=0):
    self.parameters = params
    self.lr = lr
    self.reg = reg

  def step(self):
    for param in self.parameters:
      param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])

class SGDMomentum ():
  def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
    self.l = len(params)
    self.parameters = params
    self.velocities = []
    for param in self.parameters:
      self.velocities.append(np.zeros(param['val'].shape))
    self.lr = lr
    self.rho = momentum
    self.reg = reg

  def step(self):
    for i in range(self.l):
      self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
      self.parameters[i]['val'] -= (self.lr*self.velocities[i] + self.reg*self.parameters[i]['val'])
