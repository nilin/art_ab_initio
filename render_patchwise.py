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

parser=argparse.ArgumentParser()
parser.add_argument('--makestyle',action='store_true')
args=parser.parse_args()

if args.makestyle:
    os.makedirs('pics/styles',exist_ok=True)
    picspath='pics/inspiration'
    path='pics/styles'
else:
    #picspath=json.load(open('config.json'))['render_pics_path']
    picspath='pics/cont_render'
    path=json.load(open('config.json'))['art_output_path']
os.makedirs(path,exist_ok=True)

def getpath(path='weights'):
    prevsessions=[f for f in os.listdir(path) if 'session_' in f]
    prevsessions.sort(key=lambda name:int(name.split('_')[1]))
    print('options:')
    for s in prevsessions: print(s)
    counter=int(input('\nInput session number:\nsession_'))
    return os.path.join(path,'session_{}'.format(counter))

train_res=int(json.load(open('config.json'))['train_res'])
#in_res=int(json.load(open('config.json'))['render_res_in'])
#out_res=int(json.load(open('config.json'))['render_res_out'])
#repeats=(in_res//train_res)**2


patchres=128
outres=512

runpath=getpath()
models=util.load(runpath,'models')
trainer=models['trainer']

#layerweights_bg=[1]+[0]*99
layerweights_bg=[1]+[0]*99
layerweights=[0]+[1]*99

applyfn_=lambda params,pic,bg,up: trainer.apply(
    params,pic,bg=bg,training=False,fixate=False,upscaling=up,layerweights=layerweights,border=.3
    )

applyfns=[jax.jit(partial(applyfn_,up=r)) for r in [4,2,1]]
repeats=[0,0,256]
patchsizes=[512,256,128]

dir=[fn for fn in os.listdir(picspath) if fn[0]!='.']

def savepic(img,name):
    plt.close('all')
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0,dpi=300)
    plt.close('all')


k0=rnd.PRNGKey(0)
keys=rnd.split(k0,1000)
Keys=jnp.reshape(keys,(4,250,2))


for picfile in dir:
    print('\npic: {}'.format(picfile))
    picpath=os.path.join(picspath,picfile)
    f=open(picpath,'rb')
    pic=plt.imread(f)
    pic=pic/jnp.max(pic)
    pic=jax.image.resize(pic,(outres,outres,3),jax.image.ResizeMethod.LANCZOS3)

    bgpic=jax.image.resize(pic,(patchres,patchres,3),jax.image.ResizeMethod.LANCZOS3)
    params=util.load(runpath,'params')
    _,aux=trainer.apply(params,bgpic,training=False,fixate=True,upscaling=1,layerweights=layerweights_bg)
    bg=aux['paintings']
    bg=jax.image.resize(bg,(outres,outres,3),jax.image.ResizeMethod.LANCZOS3)
    Painting=np.array(bg)

    for patchsize,fn,reps,keys in zip(patchsizes,applyfns,repeats,Keys):
        print(patchsize)

        for i,key in enumerate(keys[:reps]):
            print(i)
            x,y=rnd.choice(key,outres-patchsize+1,(2,))
            patch=Painting[x:x+patchsize,y:y+patchsize]
            picpatch=pic[x:x+patchsize,y:y+patchsize]
            patch128=jax.image.resize(patch,(patchres,patchres,3),jax.image.ResizeMethod.LANCZOS3)
            picpatch=jax.image.resize(picpatch,(patchres,patchres,3),jax.image.ResizeMethod.LANCZOS3)

            _,aux=fn(params,picpatch,jnp.array(patch))
            newlayer,recs=aux['paintings'],aux['recs']
            newpatch=colors.combine_flat(newlayer,jnp.array(patch))

            Painting[x:x+patchsize,y:y+patchsize]=newpatch

    trunc=lambda picfile:picfile.split('.')[0]
    savepic(Painting,os.path.join(path,'painting_{}.jpg'.format(trunc(picfile))))