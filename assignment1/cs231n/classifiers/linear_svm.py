import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] += -X[i].T 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg * W
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero


  # y = y.reshape(X.shape[0],1)


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  delta = 1
  pred = X.dot(W)
  num_train = X.shape[0]
  c_scores = pred[range(num_train),list(y)].reshape(num_train,1)  # num_train X 1  # spent 2 hours writing this line if intereseted read below
  """
  numpy as two types of array one is ndarray with some shape like (a,b) , (5,3,4) etc.
  and another like (a,) like (10,).
  Intially i converted y to (10,1) type because (10,) mostly creates problem and they did this time too.
  when i was trying to run this
  pred[range(num_train),list(y)]
  it was returning a N X N matrix, and when I tried using the original y which was of the shape (N,) it returns (N,) vector
  which I further convert to N,1.
  So be carefull
  """
  margins = np.maximum(0,pred - c_scores + delta) # num_train X num_classes
  margins[np.arange(num_train),list(y)] = 0 
  loss = np.sum(margins)/num_train + 0.5 * reg * np.sum(W[:,W.shape[1] - 1] * W[:,W.shape[1] - 1])  # We do not regularize bias
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask = np.zeros(pred.shape)
  mask[margins > 0] = 1
  mask[np.arange(num_train), list(y)] = -np.sum(mask, axis=1)

  dW = (X.T).dot(mask)
  dW = dW/num_train + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
