import skimage
import numpy as np
import sys
import matplotlib.pyplot as plt

def revalue_bw(x,y,strength=1.5):
    shape=x.shape
    smallshape=(32,32)
    x_=skimage.transform.resize(x,smallshape)
    y_=skimage.transform.resize(y,smallshape)
    correction=y_-x_
    correction=skimage.transform.resize(correction,shape)
    return np.maximum(0,np.minimum(255,x+strength*correction))

def revalue(x,y):
    x_=skimage.color.rgb2hsv(x)
    y_=skimage.color.rgb2hsv(y)

    newvalue=revalue_bw(x_[...,-1],y_[...,-1])
    xy_=np.concatenate([x_[...,:-1],newvalue[...,None]],axis=-1)
    return skimage.color.hsv2rgb(xy_)*1.0/255

x_path,y_path,outpath=sys.argv[1:]

def load(path):
    f=open(path,'rb')
    pic=plt.imread(f)*1.0
    return pic

x=load(x_path)
y=load(y_path)
#y=torchvision.transforms.Resize((x.shape[:-1]))(y)
y=skimage.transform.resize(y,x.shape[:-1])
img=revalue(x,y)

f=open(outpath,'wb')
plt.imshow(img)
plt.axis('off')
plt.savefig(outpath, bbox_inches='tight', pad_inches=0)