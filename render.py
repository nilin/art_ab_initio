#import main
from art import util
import sys
import jax
from jax.tree_util import tree_map
from functools import partial
import jax.numpy as jnp
import pickle
import os
import json
import matplotlib.pyplot as plt
import argparse


#picspath='projectpics/holdouts2'
picspath=json.load(open('config.json'))['render_pics_path']
outpath=json.load(open('config.json'))['art_output_path']
os.makedirs(outpath,exist_ok=True)

parser=argparse.ArgumentParser()
parser.add_argument('--contentpath',default=picspath)
#parser.add_argument('--res',default=256)
args=parser.parse_args()
picspath=args.contentpath


def getpath(reload: bool,outpath='weights'):
    prevsessions=[f for f in os.listdir(outpath) if 'session_' in f]
    prevsessions.sort(key=lambda name:int(name.split('_')[1]))

    print('options:')
    for s in prevsessions: print(s)
    counter=int(input('\nInput session number:\nsession_'))
    return os.path.join(outpath,'session_{}'.format(counter))

runpath=getpath(reload=True)
models=util.load(runpath,'models')
trainer=models['trainer']

train_res=128
in_res=512
out_res=512

#train_res=int(json.load(open('config.json'))['train_res'])
#in_res=int(json.load(open('config.json'))['render_res_in'])
#out_res=int(json.load(open('config.json'))['render_res_out'])
repeats=(in_res//train_res)**2

applyfn=jax.jit(partial(trainer.apply,training=False,upscaling=out_res//in_res,repeats=repeats))


dir=[fn for fn in os.listdir(picspath) if fn[0]!='.']

def savepic(img,name):
    plt.close('all')
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close('all')

for picfile in dir:
    print('\npic: {}'.format(picfile))
    picpath=os.path.join(picspath,picfile)
    f=open(picpath,'rb')
    pic=plt.imread(f)
    pic=pic/jnp.max(pic)

    print('\nin_res: {}'.format(in_res))

    pic=jax.image.resize(pic,(in_res,in_res,3),jax.image.ResizeMethod.LANCZOS3)

    paramfiles=[f for f in os.listdir(runpath) if f[:7]=='params_' and 'and' not in f]
    paramfiles.sort(key=lambda f:int(f.split('_')[1]))
    paramshist=[util.load(runpath,fn) for fn in paramfiles]

    trunc=lambda picfile:picfile.split('.')[0]
    savepic(pic,os.path.join(outpath,'pic_{}.jpg'.format(trunc(picfile))))
    out=[]
    for paramfile,params in zip(paramfiles,paramshist):
        print('params: {}'.format(paramfile))
        _,aux=applyfn(params,pic)
        painting,recs=aux['paintings'],aux['recs']
        out.append([pic,painting,recs])

        savepic(painting,os.path.join(outpath,'painting_{}_{}.jpg'.format(trunc(picfile),paramfile)))
        savepic(recs,os.path.join(outpath,'recs_{}_{}.jpg'.format(trunc(picfile),paramfile)))


    if 'i' in sys.argv:
        breakpoint()


