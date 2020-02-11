import numpy as np
import matplotlib.pylab as plt
import glob
import sys
from PIL import Image
from PIL import Image, ImageOps 
import os, random
from time import sleep

def getRandomStartSlice(InputSize):
    
    InputHeight=InputSize[1]
    uri=random.choice(os.listdir("images"))
    image = Image.open("images/"+uri)
        
    #convert to grayscale
    image=image.convert('L')
    image=image.resize((InputSize[0],InputSize[0]))
    
    #convert to numpy
    im=np.array(image)
    
    i=0
    xslice=im[i:i+InputHeight,:]
    xslice=xslice.reshape(xslice.shape+(1,))
    #xslice=xslice.reshape((1,) + im.shape)
    
    xslice=np.expand_dims(xslice, axis=0)
    
    return xslice,image

def concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def cropImageInputs(amount=5,InputSize=[28,28]):
    cut=int(InputSize[0]/2)
    
    cuts=[]
    
    oImgs=loadData(InputSize)
    for oI in oImgs:
        imgNP=oI[0:cut,:]
        #print("imgNP shape",imgNP.shape)
        cuts.append(imgNP)

    cuts=np.array(cuts)
    
    cuts=cuts.reshape(cuts.shape+(1,))
    
    return cuts
        
def buildFakeImages(amount=5,InputSize=[28,28]):
    cut=int(InputSize[0]/2)
    oImgs=loadData(InputSize)
    #get some random original X images
    idx = np.random.randint(0,oImgs.shape[0], amount)
    oImgs = oImgs[idx]
    #print("oImgs shape",oImgs.shape)
    
    faked=[]
    
    for oI in oImgs:
        #print("oI shape",oI.shape)
        #print("cut",cut)
        #imgNP=oI[0,cut:0,oI.shape[0]]
        imgNP=oI[0:cut,:]
        #print("imgNP shape",imgNP.shape)
        Oimg = Image.fromarray(imgNP)
        #noise
        #noise=np.random.random((cut,InputSize[0])) 
        noise=noise_generator((cut,InputSize[1]))
        
        #print("noise.shape",noise.shape)
        
        Nimg=Image.fromarray(noise,"L")
        canvas=concat_v(Oimg,Nimg)
        canvas=canvas.convert('L')
        im=np.array(canvas)
        faked.append(im)
        #canvas.show()
        #sleep(2)
        
    faked=np.array(faked)
    
    return faked

def noise_generator(shape):
    
    row,col= shape
    ch=1
    mean = 0.0
    var = 0.01
    sigma = var**0.5
    gauss = np.array(shape)
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = gauss
    return noisy#.astype('uint8')
    

def loadData(InputSize=[28,28]):
    X=[]
    
    for uri in glob.glob("images/*.jpg"):
        im = Image.open(uri)
        
        #convert to grayscale
        im=im.convert('L')
        
        im=im.resize((InputSize[0],InputSize[0]))
        
        im=np.array(im) 
        
        X.append(im)
    
    X=np.array(X)
    
    return X

def prepareData(InputSize=[800,20],predHeight=1):
    X,Y=[],[]
    
    InputHeight=InputSize[1]
    
    notifiedResized=False
    
    for uri in glob.glob("images/*.jpg"):
    #for uri in glob.glob('/content/drive/My Drive/jamonjamonjamon/images/*.jpg'):
    
        #im = plt.imread(uri)
        im = Image.open(uri)
        
        #convert to grayscale
        im=im.convert('L')

        if im.size[0]>InputSize[0]:
            if not notifiedResized:
                print(":::::::::::::::::::::::::::::::::::::::::::")
                print("Images will be resized from ",im.size[0],"to",InputSize[0],"width")
                print(":::::::::::::::::::::::::::::::::::::::::::")
                print("")
                notifiedResized=True
            im=im.resize((InputSize[0],InputSize[0])) #!important, assuming the original images will be squares
        
        #convert to numpy
        im=np.array(im) 
        
        #im=np.mean(im, axis=1) #(convert to grey)
        #print("image resized",im.shape)
        
        #im=rgb2gray(im)
        
        
        #showImage(im)
        
        
        for i in range(200):
            
            xslice=im[i:i+InputHeight,:]
            #yslice=im[i+InputHeight:(i+InputHeight)+predHeight,:]
            yslice=im[i+InputHeight:(i+InputHeight)+predHeight,:]
            
            if xslice.shape[0]<InputHeight:
                break
            else:
                #print(xslice.shape)
                #yslice=yslice.flatten(order='C')
                
                xslice=xslice.reshape(xslice.shape+(1,))
                #yslice=yslice.reshape(yslice.shape+(1,))
                
                if yslice.shape[0]>0:
                    
                    X.append(xslice)
                    Y.append(yslice.flatten())
                
        #print("")
    #sys.exit()
    X=np.array( X )
    Y=np.array( Y )
    return X,Y



"""
def prepareData(InputSize=[800,20],predHeight=1):
    X,Y=[],[]
    
    InputHeight=InputSize[1]
    
    notifiedResized=False
    
    for uri in glob.glob("images/*.jpg"):
        #im = plt.imread(uri)
        im = Image.open(uri)
     
        if im.size[0]>InputSize[0]:
            if not notifiedResized:
                print(":::::::::::::::::::::::::::::::::::::::::::")
                print("Images will be resized from ",im.size[0],"to",InputSize[0],"width")
                print(":::::::::::::::::::::::::::::::::::::::::::")
                print("")
                notifiedResized=True
            im=im.resize((InputSize[0],InputSize[0])) #!important, assuming the original images will be squares
        
        #convert to numpy
        im=np.array(im)
        #print("image resized",im.shape)
        
        for i in range(200):
            
            xslice=im[i:i+InputHeight,:,:]
            yslice=im[i+InputHeight:(i+InputHeight)+predHeight,:,:]
            
            if xslice.shape[0]<InputHeight:
                break
            else:
                #print(xslice.shape)
                X.append(xslice)
                Y.append(yslice)
                
        #print("")
    #sys.exit()
             
        
    return np.array(X),np.array(Y)

"""
def showImage(im,mode="RGB"):
    # if black and white  showImage(img[:,:,0]*255,"F")
                    
    img = Image.fromarray(im, mode)
    """
    if len(im.shape)>2:
        img = Image.fromarray(im, 'RGB')
        img.show()
    else:
        #img = Image.fromarray(im, 'RGB')
        img = Image.fromarray(im, 'L')
    """
  
    print(img)
    img.show()
  
