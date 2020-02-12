#TEST MODEL
from PIL import Image
from utils import *
from keras.models import load_model
import matplotlib.pyplot as plt
import json
from keras.models import model_from_json

#SETTINGS:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
"""
modelname="model8"
imageInputSize=[80,40]
predictionH=40
"""
modelname="generator"
imageInputSize=[32,16]
predictionH=imageInputSize[1]

predictionIter=30
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


#model = load_model('trained/'+modelname+'.h5')
print ('trained/'+modelname+'.json')
#model = model_from_json('trained/'+modelname+'.json')
with open('trained/'+modelname+'.json', 'r') as json_file:
    architecture = json.load(json_file)
    model = model_from_json(json.dumps(architecture))
    
# load weights into new model
model.load_weights('trained/'+modelname+'_weights.hdf5')

X,startImage=getRandomJamon(imageInputSize)

#print("X",X)
#showImageNormal(X[0],"RGB")

masked_X=maskimageBottom(X[0])

#print("masked_X",masked_X)
#print("masked_X shape",masked_X.shape)

masked_X=masked_X.reshape((1,)+masked_X.shape)

#print("masked_X",masked_X.shape)

#showImageNormal(masked_X,"RGB")


#cut the original image as the one used for prediction
startImage=startImage.crop((0,0,  imageInputSize[0], imageInputSize[1]))

canvas=startImage
#startImage.show()
totalH=imageInputSize[1]+predictionH

for i in range(predictionIter):
    
    predicted=model.predict(masked_X)
    
    #print (predicted)
    #sys.exit()
    #multiple rows
    #predicted=np.array(np.split(predicted,40))
    
    #predicted=predicted.reshape(predictionH,imageInputSize[0])
    
    #imgplot=plt.imshow(predicted[0])
    #plt.show()
    
    #predicted=denormalizeImage(predicted[0],-1.0, 1.0,0, 255)
    predicted=predicted[0]*255
    
    predicted = predicted.astype(np.uint8)
   
    
    p = Image.fromarray(predicted,"RGB")
    
    #p.show()
    #sleep(2)
    
    #print ("canvas",canvas)
    #print ("p",p)
    canvas=concat_v(canvas,p)
    #canvas=canvas.convert('L')
    im=np.array(canvas)
    
    xslice=im[totalH-imageInputSize[1]:totalH,:]
    
    #xslice=xslice.reshape(xslice.shape+(1,))
    
    
    
    #xslice=xslice.reshape((1,) + im.shape)
    totalH+=predictionH
    X=np.expand_dims(xslice, axis=0)
    
    masked_X=maskimageBottom(X[0])
    masked_X.resize((masked_X.shape[0]*2, masked_X.shape[1],masked_X.shape[2]))
    #print("masked_XR",masked_X)
    #print("masked_X shape",masked_X.shape)
    masked_X=masked_X.reshape((1,)+masked_X.shape)
    print("masked_X shape",masked_X.shape)
    showImage(masked_X[0])
    sleep(2)
    
    #sys.exit()
    #print("X shape",X.shape)
    #sys.exit()
    
    #canvas.show()
    #sleep(10)
    
canvas.show()