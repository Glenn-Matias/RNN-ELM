from telnetlib import XASCII
import numpy as np
from numpy.random import randn

class RNN:
  # A many-to-one Vanilla Recurrent Neural Network.

  def __init__(self, input_size, output_size, hidden_size= 1000):

    np.random.seed(0)
    np.set_printoptions(suppress=True)


    # Weights
    # self.Whh = randn(hidden_size, hidden_size)
    # self.Wxh = randn(hidden_size, input_size)
    # self.Why = randn(output_size, hidden_size)

    self.Whh = np.random.uniform(-0.5, 0.5, (hidden_size, hidden_size))
    self.Wxh = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))


    self.hidden_layer_output = np.empty((0,hidden_size), int)
    # print(self.Whh.shape)
    # print(self.Wxh.shape)

    # Biases
    # self.bh = randn(hidden_size, 1)
    # self.by = np.zeros((output_size, 1))
    # self.bh = np.random.uniform(0, 1, (1, hidden_size))
    # print(self.Wxh)
    # ELM
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
      h = np.tanh(self.Wxh @ x + self.Whh @ h)


    self.hidden_layer_output  = np.append(self.hidden_layer_output, np.array(h.T), axis=0)
    # print(self.Wxh @ x + self.Whh @ h + self.bh)
    # print(h.shape)

    # Compute the output
    # y = self.Why @ h + self.by
    # print(self.relu(y))

    # self.H = h
    # print(self.H.shape)
    return self.hidden_layer_output


  def compute_beta(self, y):

    # print(y)
    
    H = np.asmatrix(self.hidden_layer_output)

    print("After sigmoid")
    H = self.sigmoid(H)

    H_moore_penrose = np.linalg.inv(H.T * H) * H.T
    self.beta = H_moore_penrose * y

    # print("Start beta")
    # print(self.beta)
    # print("End beta")


  def predict(self, input):
    """
        Predict the results of the training process using test data
        Parameters:
        X: array-like or matrix
            Test data that will be used to determine output using ELM
        Returns:
            Predicted results or outputs from test data
    """
    # X = np.matrix(X)


    h = np.zeros((self.Whh.shape[0], 1))

    # print("Start")
    # Perform each step of the RNN
    for i, x in enumerate(input):
      h = np.tanh((self.Wxh @ x + self.Whh @ h))
    # print("End")


    # print("glenn")
    # print((self.Whh))
    # print("matias")

    # h = self.sigmoid(h) * self.beta.T 
    # print(self.Why @ (h)) 
    # y = (self.Why @ (h * self.beta.T))
    # print(y)
    # print("s")
    # print(y)
    # y = np.sum(yh)
    y = self.sigmoid(h.T) @ self.beta 
    # print(self.beta)
    # print(y.shape)

    y_sum = np.sum(y)

    print(f"{y_sum}")
    # print(self.beta)
    return y_sum
