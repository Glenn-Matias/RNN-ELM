import numpy as np
from numpy.random import randn

np.random.seed(0)
np.random.randn(1)
np.set_printoptions(suppress=True)
class RNN:
  # A many-to-one Vanilla Recurrent Neural Network.

  def __init__(self, input_size, output_size, hidden_size=3):
    # Weights
    self.Whh = randn(hidden_size, hidden_size) / 1000
    self.Wxh = randn(hidden_size, input_size) / 1000
    self.Why = randn(output_size, hidden_size) / 1000
    self.hidden_layer_output = np.empty((0,hidden_size), int)
    # print(self.Wxh.shape)
    # print(self.Whh.shape)

    # Biases
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

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
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        
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

    print(y)
    H = self.hidden_layer_output

    print("After sigmoid")
    H = self.sigmoid(H)

    print(H)
    print(H.shape)
    
    H_moore_penrose = np.linalg.inv(H.T * H) * H.T
    # self.beta = H_moore_penrose * y



    print("Start beta")
    print(self.beta)
    print("End beta")