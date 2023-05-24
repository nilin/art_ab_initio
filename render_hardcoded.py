from art import util
import sys
import jax
from jax.tree_util import tree_map
from functools import partial
import jax.numpy as jnp
import pickle
import os
from art import colors
import jax.random as rnd
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
from art import artist2

parser=argparse.ArgumentParser()
parser.add_argument('--makestyle',action='store_true')
args=parser.parse_args()

picspath='pics/imagination'
path='pics/rendered_2'
#path=json.load(open('config.json'))['art_output_path']
os.makedirs(path,exist_ok=True)

patchres=128
outres=512

painter=artist2.HardCoded(nsteps=10,ndots=10,stepsize=2,nstrokes=50,width=2)
dummyparams=painter.init(rnd.PRNGKey(0),jnp.ones((patchres,patchres,3)))
applyfn=lambda pic,key: painter.apply(dummyparams,pic,upscaling=1,key=key)
applyfn=jax.jit(applyfn)

dir=[fn for fn in os.listdir(picspath) if fn[0]!='.']

def savepic(img,name):
    plt.close('all')
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0,dpi=300)
    plt.close('all')


k0=rnd.PRNGKey(0)
keys=rnd.split(k0,1000)
#Keys=jnp.reshape(keys,(4,250,2))
merge=jax.jit(colors.merge)

for picfile in dir:
    print('\npic: {}'.format(picfile))
    picpath=os.path.join(picspath,picfile)
    f=open(picpath,'rb')
    pic=plt.imread(f)
    pic=pic/jnp.max(pic)
    pic=jax.image.resize(pic,(outres,outres,3),jax.image.ResizeMethod.LANCZOS3)

    #bgpic=jax.image.resize(pic,(patchres,patchres,3),jax.image.ResizeMethod.LANCZOS3)
    #bg=aux['paintings']
    #bg=jax.image.resize(bg,(outres,outres,3),jax.image.ResizeMethod.LANCZOS3)
    #Painting=np.array(bg)
    Painting=np.ones((outres,outres,3))

    #for patchsize,fn,reps,keys in zip(patchsizes,applyfns,repeats,Keys):
    #for keys in Keys:
    #    print(patchsize)
    patchsize=patchres

    for i,key in enumerate(keys):
        print('layer {}'.format(i))
        x,y=rnd.choice(key,outres-patchsize+1,(2,))
        patch=Painting[x:x+patchsize,y:y+patchsize]
        picpatch=pic[x:x+patchsize,y:y+patchsize]
        patch128=jax.image.resize(patch,(patchres,patchres,3),jax.image.ResizeMethod.LANCZOS3)
        picpatch=jax.image.resize(picpatch,(patchres,patchres,3),jax.image.ResizeMethod.LANCZOS3)

        colorlayers,aux=applyfn(picpatch,key)
        #newlayer,recs=aux['paintings'],aux['recs']
        newlayer=merge(colorlayers)
        newpatch=colors.combine_flat(newlayer,jnp.array(patch))

        Painting[x:x+patchsize,y:y+patchsize]=newpatch

    trunc=lambda picfile:picfile.split('.')[0]
    savepic(Painting,os.path.join(path,'painting_{}.jpg'.format(trunc(picfile))))