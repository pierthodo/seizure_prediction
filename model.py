from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from utils import *
from multiprocessing import Pool
from scipy import signal
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Lambda, Permute
from keras.layers import Convolution2D, MaxPooling3D
from keras.optimizers import SGD,Adagrad
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.engine.topology import Merge
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers import Input,merge
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers.local import LocallyConnected2D
from keras.optimizers import RMSprop
from keras.layers import TimeDistributed


def get_model(X_train_shape):
	model = Sequential()
	model.add(Permute((3,2,1),input_shape=X_train_shape))
	old_shape = model.layers[-1].output_shape

	model.add(Reshape((old_shape[1],1,old_shape[2],old_shape[3])))
	model.add(MaxPooling3D(pool_size=(1, 4, 1)))
	
	model.add(TimeDistributed(LocallyConnected2D(32,5,4,subsample=(2,2))))
	model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(LocallyConnected2D(64,5,4,subsample=(2,1))))
	model.add(TimeDistributed(Activation('relu')))
	
	old_shape = model.layers[-1].output_shape
	model.add(Reshape((old_shape[1],old_shape[2]*old_shape[3]*old_shape[4])))

	model.add(Bidirectional(LSTM(128,return_sequences= True,init ='glorot_normal')))
	model.add(Activation('relu'))
	model.add(Lambda(function=lambda x: K.mean(x, axis=1), 
				   output_shape=lambda shape: (shape[0],) + shape[2:]))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	#Try rmsprop instead here 
	sgd = Adagrad(lr = 0.001)
	model.compile(loss='binary_crossentropy',  optimizer=sgd)
	return model