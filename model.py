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


def get_model(shape):

	model = Sequential()
	model.add(Permute((3,2,1),input_shape=shape))
	old_shape = model.layers[-1].output_shape

	model.add(Reshape((old_shape[1],1,old_shape[2],old_shape[3])))
	model.add(MaxPooling3D(pool_size=(1, 4, 1)))
	
	model.add(TimeDistributed(LocallyConnected2D(32,5,4,subsample=(2,2))))
	#model.add(TimeDistributed(Activation('relu')))
	model.add(TimeDistributed(LocallyConnected2D(64,5,4,subsample=(2,1))))
	#model.add(TimeDistributed(Activation('relu')))
	
	old_shape = model.layers[-1].output_shape
	model.add(Reshape((old_shape[1],old_shape[2]*old_shape[3]*old_shape[4])))

	model.add(Bidirectional(LSTM(128,return_sequences= True,init ='glorot_normal')))
	#model.add(Activation('relu'))
	model.add(Lambda(function=lambda x: K.mean(x, axis=1), 
				   output_shape=lambda shape: (shape[0],) + shape[2:]))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	#Try rmsprop instead here 
	sgd = Adagrad(lr = 0.001)
	model.compile(loss='binary_crossentropy',  optimizer=sgd)
	return model

def get_model_id(shape):

	spec_model = Sequential()
	spec_model.add(LocallyConnected2D(32,5,4,subsample=(4,2),input_shape=(1,101,16)))
	spec_model.add(TimeDistributed(Dropout(0.25)))

	spec_model.add(LocallyConnected2D(64,5,4,subsample=(4,2)))
	spec_model.add(Dropout(0.25))



	video_input = Input(shape=(16,101,428))

	Perm = Permute((3,2,1))
	x = Perm(video_input)

	old_shape = Perm.output_shape

	x = Reshape((old_shape[1],1,old_shape[2],old_shape[3]))(x)

	x_t = TimeDistributed(spec_model)
	x = x_t(x)

	rec_layer = Bidirectional(LSTM(128,return_sequences= True,init ='glorot_normal'))
	x = rec_layer(x)

	mean_temp = Lambda(function=lambda x: K.mean(x, axis=1), 
	               output_shape=lambda shape: (shape[0],) + shape[2:])
	x = mean_temp(x)

	aux_input = Input(shape=(1,))
	x = merge([x, aux_input], mode='concat')(x)

	x = Dense(1,activation='sigmoid')(x)

	sgd = Adagrad(lr=0.001)
	model = Model(input = [video_input,aux_input],output = [x])
	model.compile(loss='binary_crossentropy',  optimizer=sgd)

	return model