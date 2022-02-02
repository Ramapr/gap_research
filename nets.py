from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense, Input, GlobalAveragePooling2D
from keras.activations import relu, elu
from keras.layers import LeakyReLU
from keras import Model
from tensorflow import convert_to_tensor
import keras.backend as K
import numpy as np
from keras.layers import SeparableConv2D, Add
#from keras.layers import ReLU  

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense, Input
from keras.layers import LeakyReLU, SeparableConv2D, GlobalAveragePooling2D, Add
from keras.layers import ReLU, Conv1D, Concatenate
from keras import Model
from tensorflow import convert_to_tensor
import keras.backend as K
import numpy as np


#####
def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true, y_pred):
  y_true = convert_to_tensor(y_true, np.float32)
  y_pred = convert_to_tensor(y_pred, np.float32)
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))
#####

