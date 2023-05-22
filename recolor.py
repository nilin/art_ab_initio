import skimage
import numpy as np
import sys
import matplotlib.pyplot as plt

def recolor(x,y):
    x_=skimage.color.rgb2hsv(x)
    y_=skimage.color.rgb2hsv(y)

    xy_=np.concatenate([y_[...,:-1],x_[...,-1:]],axis=-1)
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
img=recolor(x,y)

f=open(outpath,'wb')
plt.imshow(img)
plt.axis('off')
plt.savefig(outpath, bbox_inches='tight', pad_inches=0)