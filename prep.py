import numpy as np
import re
import os
from utils import *
import pickle
import pandas as pd
import sys

def transform(data):
    res = []
    for i in range(data.shape[1]):
        tmp_x = data[:,i]
        res.append(spectrogram(tmp_x,fs = sampling_freq,
        						window =signal.get_window(parameters[feature[0]]['windowing'],
        						parameters[feature[0]]['length']),
        						nperseg = parameters[feature[0]]['length'],
        						noverlap = parameters[feature[0]]['overlap'])[2])
    return np.array(res)

def replace_d(df,x):
    for i in range(x.shape[0]):
        df['data'][i] = x[i]
    return df

PATH = "/rap/jim-594-aa/pthodo/kaggle/data/"
data_p = 'downsample_100/'
idx_type = 'spec/val/'
nb_workers = 8

feature = ['spectrogram']
sampling_freq = int(re.search(r'\d+', data_p).group())

parameters = {}
parameters[feature[0]] = {'windowing':'hamming','length':200,'overlap':60}

directory_feature = PATH + 'cached_feature/' + 'spectrogram_100_basic'

if not os.path.exists(directory_feature):
    os.makedirs(directory_feature)
else:
    sys.exit("Data already prepared")

if not os.path.exists(PATH + 'Index/' + idx_type):
    os.makedirs(PATH + 'Index/' + idx_type)


for i in range(3):
    print "Patient" + str(i)
    X_train = pd.read_pickle(PATH+data_p+'X_train_'+str(i+1)+'.pkl')
    X_test =pd.read_pickle(PATH+data_p+'X_test_'+str(i+1)+'.pkl')


    x_f = get_feature(np.array(X_train['data']),i)
    x_f_t = get_feature(np.array(X_test['data']),i)

    X_train = replace_d(X_train,x_f)
    X_test = replace(X_test,x_f_t)
    X_train.to_pickle(directory_feature + 'X_train_'+str(i+1)+'.pkl')
    X_test.to_pickle(directory_feature + 'X_test_'+str(i+1)+'.pkl')

    for p in range(4):
    	idx = cross_val_idx(X_train,4,True)	
    	dire = PATH + 'Index/' + idx_type 
    	np.save(dire+str(i)+'_'+str(p)+'npy',np.array(idx))
    
