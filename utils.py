import numpy as np
from keras.callbacks import Callback
import keras
import keras.backend as K

class LearningRateLoggerForNadam(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    print('Optimizer:', self.model.optimizer.__class__.__name__, end=' - ')
    print(self.model.optimizer.get_config())
    self.lr_logs = []
    if not hasattr(self.model.optimizer, 'lr'):
      print('Optimizer don\'t have a "lr" attribute.')

  def on_epoch_end(self, epoch, logs={}):
    lr = self.model.optimizer.lr
    self.lr_logs.append(K.eval(lr))
    print('curent_lr: %.6f' % K.eval(lr), end=' ')

class LearningRateLogger(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    print('Optimizer:', self.model.optimizer.__class__.__name__, end=' - ')
    print(self.model.optimizer.get_config())
    self.lr_logs = []
    if not hasattr(self.model.optimizer, 'lr'):
      print('Optimizer don\'t have a "lr" attribute.')
    if not hasattr(self.model.optimizer, 'decay'):
      print('Optimizer don\'t have a "decay" attribute.')
    if not hasattr(self.model.optimizer, 'iterations'):
      print('Optimizer don\'t have a "iterations" attribute.')

  def on_epoch_end(self, epoch, logs={}):
    lr = self.model.optimizer.lr
    decay = self.model.optimizer.decay
    iterations = self.model.optimizer.iterations
    lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
    self.lr_logs.append(K.eval(lr_with_decay))
    print('curent_lr: %.6f' % K.eval(lr_with_decay), end=' ')

def get_lr_logger(model):
    if model.optimizer.__class__.__name__ == 'Nadam':
        lr_logger = LearningRateLoggerForNadam()
    else:
        lr_logger = LearningRateLogger()
    return lr_logger

def get_lr_scheduler(lr_sched, batch_sched):
    if len(lr_sched) != len(batch_sched):
        print('len(lr_sched) != len(batch_sched)')
        return None
    lr_schedule = np.ones(np.sum(batch_sched), dtype=np.float32)
    lr_schedule[:batch_sched[0]] = np.full(batch_sched[0], lr_sched[0], dtype=np.float32)
    for i in range(1, len(batch_sched)):
        lr_schedule[np.sum(batch_sched[:i]):np.sum(batch_sched[:i+1])] = np.full(batch_sched[i],
                                                                             lr_sched[i],
                                                                             dtype=np.float32)
    lr_schedule = lr_schedule.tolist()
    def scheduler(epoch):
        # print('\nScheduler set lr: %.6f'%lr_schedule[epoch])
        return lr_schedule[epoch]
    LRS = keras.callbacks.LearningRateScheduler(scheduler)
    return LRS

def get_model_checkpointer(ModelFileNameMask):
  MCheckpointer = keras.callbacks.ModelCheckpoint(ModelFileNameMask, monitor='val_loss',
                                                  verbose=0, save_best_only=False,
                                                  save_weights_only=False, mode='auto', period=1)
  return MCheckpointer


def get_callback_list(model, ModelFileNameMask):
  lr_scheduler = get_lr_scheduler(lr_sched, batch_sched)
  lr_logger = get_lr_logger(model)
  model_checkpointer = get_model_checkpointer(ModelFileNameMask)
  return [lr_scheduler, lr_logger, model_checkpointer]
