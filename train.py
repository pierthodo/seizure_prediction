import sys
sys.setrecursionlimit(100000)
import numpy as np
import os
import cPickle as pickle
import pandas as pd
from utils import *
from model import *
import sys
import csv 
import time
import argparse
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt



#########PARSE ARGUMENT ################
parser = argparse.ArgumentParser()
parser.add_argument("--submission", help="Creates a submission",action="store_true")
args = parser.parse_args()
if args.submission:
	submission = True
else:
	submission = False
########################################

#########Define variable################
PATH = "/NOBACKUP/pthodo/kaggle/data/"
PATH_RESULT = '/NOBACKUP/pthodo/kaggle/result/'
nb_epoch = 6
n_t = time.strftime("%H:%M:%S")
n_d = time.strftime("%d_%m_%Y")
num_cross_val = 1
performance_result = []
X_train,y_train,X_test,idx_list,id_set,id_set_test = load_data(PATH,submission)
########################################


########RUN ALGO########################
for cv in range(num_cross_val):
	print "Cross validation " + str(cv)
	train = []
	valid = []
	for i in range(3):
		train = train + idx_list[i][cv][0] + idx_list[i][cv][1]
		valid = valid + idx_list[i][cv][2]
		
	model = get_model(X_train.shape)
	tmp = []
	tmp_roc = 0.5
	for epoch in range(nb_epoch):

		hist = model.fit([X_train[train],id_set[train]], y_train[train], batch_size=64, nb_epoch=1,verbose= 1,validation_data=([X_train[valid],id_set[valid]],y_train[valid]),class_weight={0:1,1:10})
		
		pred = model.predict(X_train[valid])
		
		roc_1 = roc_auc_score(y_train[valid],pred)	
		loss = hist.history['loss'][0]
		
		if roc_1 > tmp_roc:
			print "Saving model"
			tmp_roc = roc_1
			model.save(PATH_RESULT+ "model/"+str(n) + "_weights.hdf5")
		
		print "Roc score at epoch number " +str(epoch) + "  :  " +  str(roc_1)
		
		tmp.append((roc_1,loss))	

	model = load_model(PATH_RESULT+ "model/"+str(n) + "_weights.hdf5")

	if submission:
		pred = model.predict([X_test,id_set_test])
		
	performance_result.append(tmp)

########################################

##########ANALYZE RESULT################

result = np.asarray(performance_result)
path_tmp = '/home/ml/pthodo/kaggle_prediction/result/' + n_d+ '/'+n_t
if not os.path.exists(path_tmp):
	os.makedirs(path_tmp)
data_m = result.mean(axis=0)   
plt.plot(data_m[:,0],label="Roc_1")
#plt.legend()
plt.savefig(path_tmp + '/' + 'performance_graph_'+'_'+n_t+'.png')
plt.close()
if submission:
	#np.save('./prediction.npy',pred)
	files = np.load('./files.npy')
	pred_tmp = []
	for i in range(len(pred)):
	    pred_tmp.append(pred[i][0])
	pred = np.array(pred_tmp)
	make_submission(pred,files)