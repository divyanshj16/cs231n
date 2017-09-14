import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    pred = X[i].dot(W)
    pred -= np.max(pred)
    pred = np.exp(pred)/np.sum(np.exp(pred))
    z = pred
    loss += -np.log(pred[y[i]])
    # print(z.shape)
    one_hot_y_i = np.zeros(num_classes).reshape(-1,1)
    one_hot_y_i[y[i]] = 1
    z = z.reshape(-1,1)
    temp = z - one_hot_y_i
    # print(X[i].shape)
    dW += X[i].reshape(-1,1).dot(temp.T)

  loss /= num_train + 0.5 * reg * np.sum(W * W)
  dW /= num_train + reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  pred = X.dot(W)
  pred = pred - np.max(pred,axis=1,keepdims=True)
  pred = np.exp(pred)/np.sum(np.exp(pred),axis=1,keepdims=True)
  loss = np.sum(-np.log(pred[range(num_train),list(y)]))/num_train + 0.5 *  reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  one_hot_y = np.zeros_like(pred)
  one_hot_y[range(num_train),list(y)] = 1
  temp = pred - one_hot_y
  dW = X.T.dot(temp)/num_train


  return loss, dW

