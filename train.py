import numpy as np
import os
import cPickle as pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
from utils import *
from multiprocessing import Pool
from scipy import signal
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adagrad
from keras.layers.recurrent import LSTM
from keras.engine.topology import Merge
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import sys

def to_np_array(X):
	if isinstance(X[0], np.ndarray):
		# return np.vstack(X)
		out = np.empty([len(X)] + list(X[0].shape), dtype=X[0].dtype)
		for i, x in enumerate(X):
			out[i] = x
		return out
def td_sum(x):
	return K.sum(x,axis=1)
def get_model():
	model = Sequential()
	model.add(Convolution2D(64, 3, 5 ,subsample = (1,2) , input_shape=(1, 101, 428) ))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(128,3, 3,subsample = (1,2)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(256,3, 3))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Convolution2D(512,3, 3))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	old_shape = model.layers[-1].output_shape
	new_shape = (old_shape[3],old_shape[1]*old_shape[2])
	model.add(Reshape(new_shape))
	model.add(Bidirectional(LSTM(128,return_sequences= True)))
	model.add(BatchNormalization())
	model.add(Lambda(function=lambda x: K.mean(x, axis=1), 
					   output_shape=lambda shape: (shape[0],) + shape[2:]))

	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	sgd = Adagrad(lr = 0.001)
	model.compile(loss='binary_crossentropy',  optimizer=sgd)


PATH = "/NOBACKUP/pthodo/kaggle/data/"
feature_p = 'cached_feature/spectrogram_100_basic/'
PATH_INDEX = 'Index/spec/val/'
PATH_RESULT = '/NOBACKUP/pthodo/kaggle/result/'
cross_patient = False
submission = False
electrode = int(sys.argv[2])
patient = int(sys.argv[1])

print "Patient " +str(patient)
X_train_pd = pd.read_pickle(PATH+feature_p+'X_train_'+str(patient)+'.pkl')
X_train = to_np_array(X_train_pd['data'])[:,electrode,:,:]
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1],
													X_train.shape[2]))
y_train = np.array(X_train_pd['Class'])
if submission:
	X_test =pd.read_pickle(PATH+feature_p+'X_test_'+str(patient)+'.pkl')
idx = np.load(PATH+PATH_INDEX + str(patient-1) + '_' + str(0))



pred_f = []
for cv in range(idx.shape[0]):
	train,valid,test = idx[cv]
	if submission:
		train = train+test
	model = get_model()
	early_stop = EarlyStopping(patience=2)
	x_t = X_train[train,:,:]
	y_t = y_train[train]
	X_valid = X_train[valid,:,:]
	y_valid = y_train[valid]
	X_test = X_train[test,:,:]
	y_test = y_train[test]
	
	model.fit(x_t, y_t, batch_size=32, nb_epoch=15,validation_data=(X_valid,y_valid),verbose= 1,callbacks=[early_stop])
	
	if submission:
		pred = model.predict(X_test)
		pred_f.append(pred)
   	else:
		pred = model.predict(X_train[test])
		pred_f.append((pred,y_train[test]))

if submission:
	np.save(PATH_RESULT + 'submission/'+str(patient)+'_'+str(electrode)+'.npy',np.array(pred_f))

else:
	np.save(PATH_RESULT + 'cross_val/'+str(patient)+'_'+str(electrode)+'.npy',np.array(pred_f))
