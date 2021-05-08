




import matplotlib.pyplot as plt
import math
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Multiply
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc

from sklearn.metrics import accuracy_score

import random
import keras

from sklearn.model_selection import train_test_split, StratifiedKFold



train_data =  np.genfromtxt('data/training_data.csv', delimiter=',')

iebe_test = np.genfromtxt('data/test_iebevishnu.csv', delimiter=',')
ipglasma_test = np.genfromtxt('data/test_ipglasma.csv', delimiter=',')


def load_data(data):
	
	X = data[1:,1:-2]
	X =  X.reshape((-1,48,15,1))
	Y = data[1:, -2]

	return X, Y


def eos_net(input_shape = (48, 15, 1), classes = 2):

  X_input = keras.Input(input_shape)



  x = layers.Conv2D(16, (8,8), padding = 'same')(X_input)
  x = layers.PReLU()(x)
  layer = layers.Dropout(0.2, input_shape= x.shape)
  x = layer(x, training = True)
  x = layers.BatchNormalization(axis = 3)(x )


  x = layers.Conv2D(32, (7,7), padding = 'same')(x)
  x = layers.PReLU()(x)
  layer = layers.Dropout(0.2, input_shape= x.shape)
  x = layer(x, training = True)
  x = layers.BatchNormalization(axis = 3)(x)
  x = layers.AveragePooling2D((2,2), padding = 'same')(x)

  x = layers.Flatten()(x)

  x = layers.Dense(128, activation = 'sigmoid')(x)
  x = layers.BatchNormalization()(x)
  layer = layers.Dropout(0.5, input_shape= x.shape)
  x = layer(x, training =  True)

  out = layers.Dense(2, activation = 'softmax')(x)

  model = Model(inputs = X_input, outputs = out, name='eos_net')

  return model


model = eos_net(input_shape = (48, 15, 1), classes = 2)

model.compile(optimizer='adam',  loss='categorical_crossentropy', metrics=['accuracy'])



def train_model(model, X_train, Y_train, X_test, Y_test):
  history = model.fit(X_train, Y_train,
          epochs = 600,
          batch_size = 64,
          use_multiprocessing=True,
          validation_data=(X_test, Y_test), verbose = 0)
          

  return history
  


def eval_model(X_test, Y_test):
  pred =  model.predict(X_test)
  pred_arg = np.argmax(pred, axis = 1)

  y_pred = np.zeros((len(pred),2))
  for i in range(len(pred)):
    y_pred[i,pred_arg[i]] = 1

  test_accuracy = accuracy_score(Y_test,y_pred)

  

  return test_accuracy

def write2file(history, test_accuracy, i):
  
  f = open("crossval_" + str(i) + "ep600run1.txt", "a")



  f.write(str(i))
  f.write('\n')
  f.write('test_accuracy')
  f.write(str(test_accuracy))

  f.write('\n')
  f.write(str(history.history))
  f.write('***********************************************************************')
  f.write('\n')


  return 


X, Y = load_data(train_data)

folds = list(StratifiedKFold(n_splits=8, shuffle=True, random_state=1).split(X, Y))

accuracy = []

for i in range(len(folds)):
  train_idx, test_idx = folds[i]
  X_train = X[train_idx]
  Y_train = keras.utils.to_categorical(Y[train_idx])

  X_test = X[test_idx]
  Y_test = keras.utils.to_categorical(Y[test_idx])

  history = train_model(model, X_train, Y_train, X_test, Y_test)
  test_accuracy = eval_model(X_test,Y_test)
  

  accuracy.append(test_accuracy)

  print(i, test_accuracy)

  write2file(history, test_accuracy, i)


f = open("crossval_ep600run1.txt", "a")

mean_accuracy = sum(accuracy)/len(accuracy)
std_dvn = np.std(np.array(accuracy))

print('accuracy = ', mean_accuracy, '+-', std_dvn)

f.write('accuracy = ') ; f.write(str(mean_accuracy));f.write( '+-'); f.write( str(std_dvn))


X, Y = load_data(iebe_test)
Y = keras.utils.to_categorical(Y)
test_accuracy = eval_model(X,Y)
print('IEBE test_accuracy =', test_accuracy)

f.write('\n')
f.write('IEBE')
f.write('iebe test accuracy:');f.write(str(test_accuracy))





X, Y = load_data(ipglasma_test)
Y = keras.utils.to_categorical(Y)

test_accuracy = eval_model(X,Y)
print('IPGLASMA test_accuracy =', test_accuracy)
f.write('\n')
f.write('IPGLASMA')
f.write('test accuracy:');f.write(str(test_accuracy))










