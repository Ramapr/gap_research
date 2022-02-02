from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.optimizers import Adam

import h5py as h5
import numpy as np
from os.path import join, exists
import pandas as pd
import re
import sys
import gc
from os import mkdir
import pickle
from os import listdir
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import f1_score, confusion_matrix, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

from importlib.machinery import SourceFileLoader
SourceFileLoader("vgg16_utils", "/data/NFP/Dev/Model/Additional/vgg16_utils.py").load_module()
from vgg16_utils import f1
SourceFileLoader("keras_utils_omilos", "/data/NFP/Dev/Model/Additional/keras_utils_omilos_cut.py").load_module()
import keras_utils_omilos
SourceFileLoader("netlib", "/home/panda/nets.py").load_module()
from netlib import *
from netlib import f1_m
from tensorflow import convert_to_tensor

# PARAMS
#%%
path2dset = "/data/dset"
_path = '/data/panda/test/bn' 

part = 'list_verh_'
tr = 'train.hdf5'
ts =  'test.hdf5'

inp_shape = (192, 256, 3)
n_class = 4 
LR = .005
EPOCH = 30 # 30
BS = 32
batch_sched = [10, 10, 5, 5] #, 16, 16]
lr_sched = [.005, .001, .0005, .0001] #, .00001]
N = 5

## 'name' : func()
dir_ = {'gap_v1': net_gap_var1,
        'gap_v2': net_gap_var2, 
        'gap_v3': net_gap_var3}

#clmn = ['name', 'n_iter', 'file', 'scikit_f1', 'cust_f1', 'custm_f1', 'loss'] #### FIX
clmn = ['name', 'n_iter', 'file', 'scikit_f1', 'cust_f1', 'AP', 'roc_auc', 'loss', 'prec', 'recall'] #### FIX

#%%
tr_file = h5.File(join(path2dset, part + tr), 'r')
tr_img = tr_file['Aug_array'][...] / 255.
tr_y = tr_file['TAG_Aug_array'][...]
tr_file.close()

ts_file = h5.File(join(path2dset, part + ts), 'r')
ts_img = ts_file['Aug_array'][...] / 255.
ts_y = ts_file['TAG_Aug_array'][...]
ts_file.close()

mean_img = np.mean(tr_img, axis=0)

X_train, X_test = tr_img - mean_img, ts_img - mean_img
y_train, y_test = tr_y, ts_y

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
#%%
for nm, dl in dir_.items():
  # create folder
  if not exists(join(_path, nm)):
    mkdir(join(_path, nm))
  model_path = join(_path, nm)
  
  log = pd.DataFrame([], columns=clmn)
  mdl_iter = 0
  
  for n in range(N):
    K.clear_session()        
    mdl = dl(inp_shape) #, n_class)
    opt = Adam(lr=LR, decay=1e-6)
    
    mdl.compile(optimizer=opt, 
                loss=categorical_crossentropy, 
                metrics=[f1, f1_m])
    
    net_name = '{0}_{1}'.format(nm, n)
    name = net_name + ".{epoch:03d}-{loss:.4f}-{val_loss:.4f}-{f1:.4f}-{val_f1:.4f}-{f1_m:.4f}-{val_f1_m:.4f}.h5"
    FileName = join(model_path, name)
    callback_list = get_callback_list(mdl, FileName)    
    model_hist = mdl.fit(X_train,
                         y_train,
                         epochs=EPOCH,
                         batch_size=BS,
                         validation_data=(X_test, y_test),
                         verbose=2,
                         shuffle=True,
                         callbacks=callback_list)

    with open(join(model_path, net_name + '_hist.pkl'), 'wb') as f:
      pickle.dump(model_hist.history, f)
    del mdl
    gc.collect()
    ########### EVAL
    files = list(filter(lambda x: True if x.endswith('.h5') and x.startswith(net_name) else False, listdir(model_path)))
    f1v = np.asarray([float(f.split('-')[-1][:-3]) for f in files])
    top5 = f1v.argsort()[-pred_n:]
    f2pred = np.asarray(files)[top5]
    for f in f2pred:
      K.clear_session()
      model = load_model(join(model_path, f), custom_objects={'f1':f1, 'f1_m':f1_m})
      y_pred = np.round(model.predict(X_test)).astype(np.uint8)
      del model
      gc.collect()
      print(f)
      #print(model.summary())
#      sys.stdout.flush()
#      scf1 = f1_score(y_test, y_pred, average='weighted')
#      #sys.stdout.flush()    
#      cf1 = K.get_value(f1(y_test, y_pred))
#      #sys.stdout.flush()
#      f1m = K.get_value(f1_m(y_test, y_pred))
#      #sys.stdout.flush()
#            
#      print("sc - {0}\ncu - {1}\ncum - {2}".format(scf1, cf1, f1m))
#      sys.stdout.flush()
      scf1 = f1_score(y_test, y_pred, average='weighted')
      cf1 = K.get_value(f1(y_test, y_pred))
      AP = average_precision_score(y_test, y_pred)
      roc_auc = roc_auc_score(y_test, y_pred)
      ca = K.get_value(categorical_crossentropy(convert_to_tensor(y_test, np.float32),
                                                convert_to_tensor(y_pred, np.float32)))
      cc = np.mean(ca)
      pre = precision_score(y_test, y_pred, average='weighted')
      rec = recall_score(y_test, y_pred, average='weighted')      
      print("sc - {0}\ncu - {1}".format(scf1, cf1)) #, f1m))
      sys.stdout.flush()
      pred = np.argmax(y_pred, 1)
      ref = np.argmax(y_test, 1)
      cm = confusion_matrix(ref, pred)
      np.save(join(model_path, f[:-3] + '_cm.npy' ), cm)
      log.loc[mdl_iter] = [nm, n, f, scf1, cf1, AP, roc_auc, cc, pre, rec]
      mdl_iter += 1
#      pred = np.argmax(y_pred, 1)
#      ref = np.argmax(y_test, 1)
#      cm = confusion_matrix(ref, pred)
#      np.save(join(model_path, f[:-3] + '_cm.npy' ), cm)
#      log.loc[mdl_iter] = [nm, n, f, scf1, cf1, f1m]
#      mdl_iter += 1
      log.to_hdf(join(model_path, 'log{0}.h5'.format(mdl_iter)), key='df', mode='w')

