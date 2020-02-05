#TEST MODEL
from PIL import Image
from utils import *
from keras.models import load_model

#SETTINGS:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
"""
modelname="model8"
imageInputSize=[80,40]
predictionH=40
"""
modelname="model9"
imageInputSize=[80,40]
predictionH=1

predictionIter=200
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


model = load_model('trained/'+modelname+'.h5')

X,startImage=getRandomStartSlice(imageInputSize)

#cut the original image as the one used for prediction
startImage=startImage.crop((0,0,  imageInputSize[0], imageInputSize[1]))

canvas=startImage
#startImage.show()
totalH=imageInputSize[1]+predictionH

for i in range(predictionIter):
    
    predicted=model.predict(X)
    
    #multiple rows
    #predicted=np.array(np.split(predicted,40))
    predicted=predicted.reshape(predictionH,imageInputSize[0])
    
    p = Image.fromarray(predicted)
    
    canvas=concat_v(canvas,p)
    canvas=canvas.convert('L')
    im=np.array(canvas)
    
    xslice=im[totalH-imageInputSize[1]:totalH,:]
    
    xslice=xslice.reshape(xslice.shape+(1,))
    #xslice=xslice.reshape((1,) + im.shape)
    totalH+=predictionH
    X=np.expand_dims(xslice, axis=0)
    #canvas.show()
    
canvas.show()