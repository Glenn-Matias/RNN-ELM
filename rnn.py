from telnetlib import XASCII
import numpy as np
from numpy.random import randn
np.random.seed(0)
np.set_printoptions(suppress=True)

class RNN:
  # A many-to-one Vanilla Recurrent Neural Network.

  def __init__(self, input_size, output_size, hidden_size):
    # Weights
    self.Whh = randn(hidden_size, hidden_size)
    self.Wxh = randn(hidden_size, input_size)
    # self.Why = randn(output_size, hidden_size)

    # self.Whh = np.random.uniform(-0.5, 0.5, (hidden_size, hidden_size))
    # self.Wxh = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))

    self.bh = randn(hidden_size, 1)
    self.hidden_layer_output = np.empty((0,hidden_size), int)
    # print(self.Wxh.shape)

    # Biases
    self.beta = 0


  def sigmoid(self, x):
    """
        Sigmoid activation function
        
        Parameters:
        x: array-like or matrix
            The value that the activation output will look for
        Returns:      
            The results of activation using sigmoid function
    """
    return 1 / (1 + np.exp(-1 * x))


  def forward(self, inputs):
    '''
    Perform a forward pass of the RNN using the given inputs.
    Returns the final output and hidden state.
    - inputs is an array of one hot vectors with shape (input_size, 1).
    '''


    h = np.zeros((self.Whh.shape[0], 1))

    # Perform each step of the RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)


    self.hidden_layer_output  = np.append(self.hidden_layer_output, np.array(h.T), axis=0)

    return self.hidden_layer_output


  def compute_beta(self, y):
    
    H = np.asmatrix(self.hidden_layer_output)

    H_moore_penrose = np.linalg.inv(H.T * H) * H.T
    self.beta = H_moore_penrose * y


  def predict(self, input):
    h = np.zeros((self.Whh.shape[0], 1))

    for i, x in enumerate(input):
      h = np.tanh((self.Wxh @ x + self.Whh @ h + self.bh))

    y = h.T @ self.beta 
    # y = h.T.dot(self.beta)

    y_sum = np.sum(y)

    return y_sum
