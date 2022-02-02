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

def net_gap(input_shape):
  inp = Input(input_shape)
######## 1 
  y1 = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  #y1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  y1 = BatchNormalization()(y1)
  #y1 = LeakyReLU()(y1)

  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 2
  y1 = SeparableConv2D(48, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = SeparableConv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 3
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = SeparableConv2D(16, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
####### 4 
  y1 = SeparableConv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = GlobalAveragePooling2D()(y1)

#######  
  out = Dense(4, kernel_initializer='normal', activation='softmax')(y1)
  mdl = Model(inp, out)
  return mdl

def net_flat(input_shape):
  inp = Input(input_shape)
######## 1 
  y1 = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  #y1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  y1 = BatchNormalization()(y1)
  #y1 = LeakyReLU()(y1)

  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 2
  y1 = SeparableConv2D(48, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = SeparableConv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 3
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = SeparableConv2D(16, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
####### 4 
  y1 = SeparableConv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  #y1 = GlobalAveragePooling2D()(y1)
  y1 = Flatten()(y1)
  y1 = Dropout(0.3)(y1)
#######  
  out = Dense(4, kernel_initializer='normal', activation='softmax')(y1)
  mdl = Model(inp, out)
  return mdl

#%%



def net_gap_conv(input_shape):
  inp = Input(input_shape)
######## 1 
  y1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  y1 = BatchNormalization()(y1)
  
  y1 = Conv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 2
  y1 = Conv2D(48, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = Conv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 3
  y1 = Conv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = Conv2D(16, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
####### 4 
  y1 = Conv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = GlobalAveragePooling2D()(y1)
#######  
  out = Dense(4, kernel_initializer='normal', activation='softmax')(y1)
  mdl = Model(inp, out)
  return mdl

def net_flat_conv(input_shape):
  inp = Input(input_shape)
######## 1 
  y1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  y1 = BatchNormalization()(y1)

  y1 = Conv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 2
  y1 = Conv2D(48, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = Conv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 3
  y1 = Conv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = Conv2D(16, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
####### 4 
  y1 = Conv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  #y1 = GlobalAveragePooling2D()(y1)
  y1 = Flatten()(y1)
  y1 = Dropout(0.3)(y1)
#######  
  out = Dense(4, kernel_initializer='normal', activation='softmax')(y1)
  mdl = Model(inp, out)
  return mdl

#%%

def net_gap_test(input_shape):
  inp = Input(input_shape)
######## 1 
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(inp)
  y1 = BatchNormalization()(y1)

  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 2
  y1 = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 3
  y1 = SeparableConv2D(128, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)

  y1 = SeparableConv2D(128, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
####### 4 
  y1 = SeparableConv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = GlobalAveragePooling2D()(y1)
#######  
  out = Dense(4, kernel_initializer='normal', activation='softmax')(y1)
  mdl = Model(inp, out)
  return mdl

#%%
def net_gap_strd(input_shape):
  inp = Input(input_shape)
######## 1 
  y1 = SeparableConv2D(64, (3, 3), strides=(2, 2), padding='valid', activation='relu')(inp)
  #y1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  #y1 = SeparableConv2D(32, (3, 3), strides=(2, 2), padding='valid', activation='relu')(inp)
  y1 = BatchNormalization()(y1)
  #y1 = LeakyReLU()(y1)
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
####### 2
  y1 = SeparableConv2D(48, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = SeparableConv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
####### 3
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = SeparableConv2D(16, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
####### 4 
  y1 = SeparableConv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = GlobalAveragePooling2D()(y1)
#######  
  out = Dense(4, kernel_initializer='normal', activation='softmax')(y1)
  mdl = Model(inp, out)
  return mdl




#%%
def net_flat_elu(inp_shape):
  #inp_shape = (192, 256, 3)
  inputs = Input(inp_shape)
##### block 1
  y1 = Conv2D(64, (3, 3), padding='valid', bias=False)(inputs)
  y1 = BatchNormalization()(y1)
  y1 = elu(y1)
  #print(y1.shape)

  y1 = Conv2D(32, (3, 3), padding='valid', bias=False)(y1)
  y1 = BatchNormalization()(y1)
  y1 = elu(y1)
  #print(y1.shape)

  y1 = MaxPooling2D((2, 2))(y1)

##### block 2
  y1 = Conv2D(48, (3, 3), padding='valid', bias=False)(y1)
  y1 = BatchNormalization()(y1)
  y1 = elu(y1)

  y1 = Conv2D(24, (3, 3), padding='valid', bias=False)(y1)
  y1 = BatchNormalization()(y1)
  y1 = elu(y1)
  y1 = MaxPooling2D((2, 2))(y1)

##### block 3
  y1 = Conv2D(16, (3, 3), padding='valid', bias=False)(y1)
  y1 = BatchNormalization()(y1)
  y1 = elu(y1)
  y1 = MaxPooling2D((2, 2))(y1)

  y1 = Conv2D(16, (1, 1), padding='valid', activation='relu')(y1)
  x = Flatten()(y1)
  x = Dropout(0.3)(x)
  out = Dense(4, kernel_initializer='normal', activation='softmax')(x)
  mdl = Model(inputs, out)
  return mdl

#%%  
def net_v3(inp_shape):
  inputs = Input(inp_shape)
  y1 = Conv2D(64, (3, 3), padding='valid', activation=elu)(inputs)
  y1 = BatchNormalization()(y1)

  y1 = Conv2D(32, (3, 3), padding='valid', activation=elu)(y1)
  y1 = BatchNormalization()(y1)

  y1 = MaxPooling2D((2, 2))(y1)

  y1 = Conv2D(48, (3, 3), padding='valid', activation=elu)(y1)
  y1 = BatchNormalization()(y1)

  y1 = Conv2D(24, (3, 3), padding='valid', activation=elu)(y1)
  y1 = BatchNormalization()(y1)

  y1 = MaxPooling2D((2, 2))(y1)

  y1 = Conv2D(16, (3, 3), padding='valid', activation=elu)(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)


  y1 = Conv2D(16, (1, 1), padding='valid', activation=elu)(y1)

  x = Flatten()(y1)
  x = Dropout(0.3)(x)
  out = Dense(4, kernel_initializer='normal', activation='softmax')(x)
  mdl = Model(inputs, out)
  return mdl


def net_t(input_shape):
  inp = Input(input_shape)
######## 1 
  y1 = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  #y1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  y1 = BatchNormalization()(y1)
  #y1 = LeakyReLU()(y1)
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 2
  y1 = SeparableConv2D(48, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = SeparableConv2D(24, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)

####### 3
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
  #y1 = MaxPooling2D((2, 2))(y1)
  y1 = SeparableConv2D(16, (3, 3), padding='valid', activation='relu')(y1)
  y1 = GlobalAveragePooling2D()(y1)
#######  
  out = Dense(4, kernel_initializer='normal', activation='softmax')(y1)
  mdl = Model(inp, out)
  return mdl #decoded

#%%
  
def net_r(input_shape):
#  input_shape = (192, 256, 3)
  inp = Input(input_shape)
######## 1 
  #y1 = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(inp)
  y1 = Conv2D(16, (3, 3), padding='valid', activation='relu')(inp)
  y1 = BatchNormalization()(y1)
#  print('conv1', y1.shape)  
  # here 1  
  y1 = SeparableConv2D(16, (3, 3), padding='valid', activation='relu')(y1)
#  print('conv02', y1.shape)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
#  print('conv2', y1.shape)  
  x1 = Conv2D(1, (5, 5), padding='valid', strides=2, activation='relu')(y1)
  x2 = Conv2D(32, (1, 1), padding='valid', activation='relu')(x1)
#  x2.shape  
####### 2
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
  y1 = BatchNormalization()(y1)
#  print('conv3', y1.shape)
  y1 = SeparableConv2D(32, (3, 3), padding='valid', activation='relu')(y1)
#  print('conv04', y1.shape)
  y1 = BatchNormalization()(y1)
  y1 = MaxPooling2D((2, 2))(y1)
#  print('conv4', y1.shape)
  ad = Add()([x2, y1])
####### 3
  y1 = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(ad)
  y1 = BatchNormalization()(y1)
#  print('conv5', y1.shape)
  #y1 = MaxPooling2D((2, 2))(y1)
  y1 = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(y1)
#  print('conv06', y1.shape)
  y1 = BatchNormalization()(y1)
  
  y1 = GlobalAveragePooling2D()(y1)
#######  
  out = Dense(4, kernel_initializer='normal', activation='softmax')(y1)
  mdl = Model(inp, out)
  return mdl #decoded



