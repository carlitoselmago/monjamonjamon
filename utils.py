import numpy as np
import matplotlib.pylab as plt
import glob
import sys
from PIL import Image
from PIL import Image, ImageOps 
import os, random



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
def showImage(im):
    if len(im.shape)>2:
        img = Image.fromarray(im, 'RGB')
        img.show()
    else:
        #img = Image.fromarray(im, 'RGB')
        img = Image.fromarray(im, 'L')
    #img.save('my.png')
    print(img)
    img.show()
    """
    Helper function to plot an image.
    """
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y/x) * h
    plt.figure(figsize=(w,h))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')
    plt.show()
    """
