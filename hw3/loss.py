
from __future__ import print_function

import numpy as np

import activation

# Negative Log Likelihood Loss
def NLLLoss (Y_pred, Y_true):
  loss = 0.0
  N = Y_pred.shape[0]
  M = np.sum(Y_pred*Y_true, axis=1)
  for e in M:
    #print(e)
    if e == 0:
      loss += 500
    else:
      loss += -np.log(e)
  return loss/N

class CrossEntropyLoss ():
  def __init__(self):
    pass

  def get (self, Y_pred, Y_true):
    N = Y_pred.shape[0]
    softmax = activation.Softmax()
    prob = softmax._forward(Y_pred)
    loss = NLLLoss (prob, Y_true)
    Y_serial = np.argmax(Y_true, axis=1)
    dout = prob.copy()
    dout[np.arange(N), Y_serial] -= 1
    return loss, dout

class SoftmaxLoss ():
  def __init__(self):
    pass

  def get (self, Y_pred, Y_true):
    N = Y_pred.shape[0]
    loss = NLLLoss(Y_pred, Y_true)
    Y_serial = np.argmax(Y_true, axis=1)
    dout = Y_pred.copy()
    dout[np.arange(N), Y_serial] -= 1
    return loss, dout
