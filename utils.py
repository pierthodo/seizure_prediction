import numpy as np
from multiprocessing import Pool
from scipy.signal import spectrogram
from scipy import signal
import pandas as pd
import os
import sys
import csv
import pickle
import time
def cross_val_idx(df,cv,validation):
    h = np.array(df['Hour'])
    cl = np.array(df['Class'])
    nb_1 = int(np.sum(cl)/6)+1
    nb_0 = int((cl.shape[0] - np.sum(cl))/6)+1
    idx_1 = np.array_split(np.random.permutation(np.arange(nb_1)),cv)
    idx_0 = np.array_split(np.random.permutation(np.arange(nb_0)),cv)
    idx = []
    for p in range(cv):
        train = []
        test = []
        if validation:
            val = []
        for i in range(h.shape[0]):  
            if cl[i] == 0:
                if (h[i] in idx_0[p]):
                    test.append(i)
                else:
                    if (h[i] in idx_0[(p+1)%cv]) and validation:
                        val.append(i)
                    else:                        
                        train.append(i)
            else:
                if (h[i] in idx_1[p]):
                    test.append(i)
                else:
                    if (h[i] in idx_1[(p+1)%cv]) and validation:
                        val.append(i)
                    else:                        
                        train.append(i)
        if validation:
            idx.append((train,val,test))    
        else:
            idx.append((train,test))    
    return idx

def to_np_array(X):
    if isinstance(X[0], np.ndarray):
        # return np.vstack(X)
        out = np.empty([len(X)] + list(X[0].shape), dtype=X[0].dtype)
        for i, x in enumerate(X):
            out[i] = x
        return out
def td_sum(x):
    return K.sum(x,axis=1)

def make_submission(pred,files):
    if not os.path.exists('./../submission/' + time.strftime("%d_%m_%Y")):
        os.makedirs('./../submission/' + time.strftime("%d_%m_%Y"))
    with open('./../submission/' +time.strftime("%d_%m_%Y")+ '/'+time.strftime("%H:%M:%S")+ '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["File","Class"])
        for i,f in enumerate(files):
            wr.writerow([f,pred[i]])   

def add(idx,offset):
    for i in range(4):
        train,test,valid = idx[i]
        train = [x+offset for x in train]
        test = [x+offset for x in test]
        valid = [x+offset for x in valid]
        idx[i] = (train,test,valid)
    return idx

def to_np_array(X):
    if isinstance(X[0], np.ndarray):
        # return np.vstack(X)
        out = np.empty([len(X)] + list(X[0].shape), dtype=X[0].dtype)
        for i, x in enumerate(X):
            out[i] = x
        return out

    return np.array(X) 

def load_data(PATH,submission):
    feature_p = 'cached_feature/spectrogram_100_basic/'
    PATH_INDEX = 'Index/spec/val/'
    idx_list = []
    size_set = 0
    id_set = []
    X_test_c = []
    id_set_test = []
    for i in range(3):
        patient = i + 1
        print "Loading patient number : "+str(patient)
        X_train_pd = pd.read_pickle(PATH+feature_p+'X_train_'+str(patient)+'.pkl')
        X_train = to_np_array(X_train_pd['data'])
        y_train = np.array(X_train_pd['Class'])
        idx = np.load(PATH+PATH_INDEX + str(i) + '_' + str(0)+'.npy')
        idx = add(idx,size_set)
        size_set += X_train.shape[0]
        id_set = id_set + list(np.ones((X_train.shape[0]))*patient)
        print "The size of the set is now " + str(size_set)
        del X_train_pd
        print submission
        if submission == True:
            print "Loading test data"
            X_test_pd =pd.read_pickle(PATH+feature_p+'X_test_'+str(patient)+'.pkl')
            X_test = to_np_array(X_test_pd['data'])
            id_set_test = id_set_test + list(np.ones((X_test.shape[0]))*patient)
            del X_test_pd
        if i == 0:
            X_train_c = X_train
            y_train_c = y_train
            if submission:
                X_test_c = X_test
        else:
            X_train_c = np.concatenate((X_train_c,X_train))
            y_train_c = np.concatenate((y_train_c,y_train))
            if submission:
                X_test_c = np.concatenate((X_test_c,X_test))
        idx_list.append(idx)

    id_set_test = np.array(id_set_test)
    id_set = np.array(id_set)

    return X_train_c,y_train_c,X_test_c,idx_list,np.array(id_set),np.array(id_set_test)