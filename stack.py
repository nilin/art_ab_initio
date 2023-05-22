import skimage
import numpy as np
import sys
import matplotlib.pyplot as plt

def block(x,y,z,w):
    out=np.concatenate([
    np.concatenate([x[:,:],y[:,::-1]],axis=1),
    np.concatenate([z[::-1,:],w[::-1,::-1]],axis=1)
    ])
    return out/255
    
*paths,outpath=sys.argv[1:]
if len(paths)==1:
    paths=paths*4
if len(paths)==2:
    paths=paths*2

def load(path):
    f=open(path,'rb')
    pic=plt.imread(f)*1.0
    return pic

blocks=[load(path) for path in paths]
img=block(*blocks)
#x=load(x_path)
#y=load(y_path)
##y=torchvision.transforms.Resize((x.shape[:-1]))(y)
#y=skimage.transform.resize(y,x.shape[:-1])
#img=recolor(x,y)

f=open(outpath,'wb')
plt.imshow(img)
plt.axis('off')
plt.savefig(outpath, bbox_inches='tight', pad_inches=0)