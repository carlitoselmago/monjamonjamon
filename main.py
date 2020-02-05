import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import pandas as pd
import os
from utils import *
from keras.models import load_model

#SETTINGS
from models.model9 import *

loss='mse'#'mean_squared_error' # Since the metric is MSE/RMSE, no change
optimizer = 'rmsprop'#rmsprop'     # Recommended optimizer for RNN, no change
batch_size = 10
nb_epoch =  300

#input data
imageInputSize=[80,40]#[400,10] #X,Y .original is 800x800 if lower, image will be resized
outputSize=[imageInputSize[0],1]
channels=1 #rgb or black and white(1)

import numpy as np
import matplotlib.pylab as plt
import glob
import sys
from PIL import Image

#PREPARE DATA

#X,Y=prepareData(imageInputSize[1],outputSize[1])
X,Y=prepareData(imageInputSize,outputSize[1])

#inputShape=(imageInputSize[1],imageInputSize[0],channels)
inputShape=(imageInputSize[1],imageInputSize[0],channels)

print("X",X.shape)
print("Y",Y.shape)

if os.path.isfile('trained/'+modelname+'.h5'):
    model = load_model('trained/'+modelname+'.h5')
else:
    pass
    model=createModel(inputShape,loss,optimizer)

#train
history=model.fit(X,Y,batch_size=batch_size,epochs=nb_epoch,validation_split=0.2) 

model.save('trained/'+modelname+'.h5')

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

  