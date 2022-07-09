import numpy as np
import random
np.random.seed(0)
random.seed(0)

from rnn import RNN
from data import train_data, test_data
from collections import OrderedDict
import pandas as pd

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

df = pd.read_csv('data/full.csv')

data_size_each = 300
df_positive = df.head(data_size_each)
df_negative = df.tail(data_size_each)
df = pd.concat([df_positive, df_negative])
df['class'] = df['label']
df['text'] = df['article']
df = df[['text', 'class']]

train, test = train_test_split(df, test_size=0.2)


df_dict =  OrderedDict(dict(df.values))
train_data = OrderedDict(dict(train.values))
test_data =  OrderedDict(dict(test.values))

# Create the vocabulary.
vocab = list(dict.fromkeys([w for text in df_dict.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)

# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }

idx_to_word = { i: w for i, w in enumerate(vocab) }
# print(word_to_idx['good'])
# print(idx_to_word[0])
print("done word/idx")
def createInputs(text):
  '''
  Returns an array of one-hot vectors representing the words in the input text string.
  - text is a string
  - Each one-hot vector has shape (vocab_size, 1)
  '''
  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs

def softmax(xs):
  # Applies the Softmax Function to the input array.
  return np.exp(xs) / sum(np.exp(xs))

# Initialize our RNN!
rnn = RNN(vocab_size, 1)

def processData(data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping text to True or False.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  # random.shuffle(items)
  loss = 0
  num_correct = 0
  accumulated_target = np.empty((0,1), int)
  for x, y in items:
    inputs = createInputs(x)

    # for i, x in enumerate(inputs):
    #   print("*")
    #   print(x)
    # Forward
    # out, _ = rnn.forward(inputs, y)
    print(accumulated_target.shape)
    hidden_layer_output = rnn.forward(inputs)
    accumulated_target  = np.vstack([accumulated_target, np.array([int(y)])])
    # accumulated_predict += [probs]
  print("Hidden layer output")
  # print(hidden_layer_output)

  rnn.compute_beta(accumulated_target)
    # probs = softmax(probs)
    # print(probs)
    # print(np.argmax(probs))
    # break
  
    # # Calculate loss / accuracy
    # loss -= np.log(probs[target])
    # num_correct += int(np.argmax(probs) == target)

    # if backprop:
    #   # Build dL/dy
    #   d_L_d_y = probs
    #   d_L_d_y[target] -= 1

    #   # Backward
    #   rnn.backprop(d_L_d_y)

  # return loss / len(data), num_correct / len(data)
  return None, None

# train_loss, train_acc = processData(train_data)
processData(train_data)
loss = 0
num_correct = 0



items = list(test_data.items())
for x, y in items:
  inputs = createInputs(x)
  pred_proba = rnn.predict(inputs)

  y_pred = (pred_proba > 0.5).astype(int)
  print(f"{pred_proba} -> {y_pred}")

  if y_pred == int(y): num_correct+=1

print(f"Total right: {num_correct} / {len(items)}")






  # num_correct += int(np.argmax(probs) == target)


# print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

# test_loss, test_acc = processData(test_data, backprop=False)
# print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))


# # Training loop
# for epoch in range(1000):
#   train_loss, train_acc = processData(train_data)

#   if epoch % 100 == 99:
#     print('--- Epoch %d' % (epoch + 1))
#     print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

#     test_loss, test_acc = processData(test_data, backprop=False)
#     print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

# default
# Train:  Loss 0.693 | Accuracy: 0.466
# Test:   Loss 0.693 | Accuracy: 0.450