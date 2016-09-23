import numpy as np
from multiprocessing import Pool
from scipy.signal import spectrogram
from scipy import signal

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


def get_feature(x,patient_id):
    p = Pool(nb_workers)
    f_x = p.map(transform,x)   
    p.close()
    f_x = to_np_array(f_x)
    print f_x.shape
    return f_x

