import pandas as pd
import numpy as np
import os
import re
import keras
from sklearn.model_selection import train_test_split
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

model = keras.models.load_model('trained_model.hdf5')
#Show the model architecture
model.summary()

# Extracting the simulation parameters of the test set trajectories 
path, dirs, files = next(os.walk("/export/home/arif/MLatom/fmo_strategies/fmo_data/test_data/2760")) 
file_count = len(files)
print("number of files = ", file_count)
# create empty list
df_list = []
gamma = np.zeros((file_count), dtype=float)
lamb = np.zeros((file_count), dtype=float)
temp = np.zeros((file_count), dtype=float)
initial = np.zeros((file_count), dtype=float)

for i in range(file_count):
    x = re.split(r'_', files[i])
    y = re.split(r'-', x[1])
    initial[i] = y[1]
    y = re.split(r'-', x[2]) # extracting value of gamma
    gamma[i] = y[1]
    y = re.split(r'-', x[3]) # extract value of lambda
    lamb[i] = y[1]
    y = re.split(r'-', x[4])
    x = re.split(r'.n', y[1]) # extract value of temperature
    temp[i] = x[0]

nsteps = 801
tt = np.zeros(nsteps, dtype=float)
tt[0:501] = np.arange(0,501) * 5.0   # 5 fs step
tt[501:nsteps] = np.arange(505,2005, 5) * 5.0  # 25 fs
x = np.zeros((1, 4), dtype=float)
y = np.zeros((nsteps, 49), dtype=float)
y1 = np.zeros((nsteps*49, 1), dtype=float)
y2 = np.zeros((49, 1), dtype=float)

#
ti = time.time()
for f in range(0, file_count):
    if initial[f] == 1:
        init_label = 0
    else:
        init_label = 1
    x[0,0] = init_label
    x[0,1] = gamma[f]/300   # normalize 
    x[0,2] = lamb[f]/310
    x[0,3] = temp[f]/310
    x_pred = x[0,:]
    x_pred = x_pred.reshape(1, x.shape[1],1) # reshape the input 
    yhat = model.predict(x_pred, verbose=0)
    y1[:,0] = yhat
    a = 0; b = 49;
    for i in range(0, nsteps):
        y[i,:] = y1[a:b,0]
        a = a + 49
        b = b + 49
    filename = "pred_init-" + str(int(initial[f])) + "_gamma-" + str(gamma[f]) + "_lambda-" + str(lamb[f]) + "_temp-" + str(temp[f]) + ".dat"
    np.savetxt(filename, np.c_['-1',tt, y])
print("time spent =", time.time() - ti)
                                            
