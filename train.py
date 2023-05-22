from art.bookkeeping import prep
from art.bookkeeping import session
import jax.numpy as jnp
import matplotlib.pyplot as plt
from art import models as mod
from art import artist
from art.trainer import Trainer, BWTrainer
import sys
from typing import List, Tuple, Dict, Callable
from flax.training.train_state import TrainState
from collections import deque
import argparse
import copy
from art import geometry
import flax
from art import colors
from jax.tree_util import tree_flatten,tree_unflatten
import optax, jax
import json
import flax
import math
import jax.random as rnd
import time
import flax.linen as nn
import numpy as np
from art.bookkeeping.session import nextkey
from jax.tree_util import tree_map
import pickle
from functools import partial
from art import util
import os

cfg=json.load(open('config.json'))

parser=argparse.ArgumentParser()
parser.add_argument('--img_path_train',default=cfg['img_path_train'])
parser.add_argument('--reload_img_train',action='store_true')
parser.add_argument('--bw',action='store_true')
parser.add_argument('--reload',action='store_true')
parser.add_argument('--res',type=int,default=int(cfg['train_res']))
args=parser.parse_args()
res=args.res
reload=args.reload


class Loader:
    nx=1
    ny=1

    def __init__(self,picspath):
        print('collecting train data')
        filenames=[fn for fn in os.listdir(picspath) if fn[0]!='.']
        pics=[]
        for i,fn in enumerate(filenames):
            print('{}/{}'.format(i+1,len(filenames)))
            filepath=os.path.join(picspath,fn)
            f=open(filepath,'rb')
            pic=plt.imread(f)*1.0
            pic=pic/jnp.max(pic)
            #pics=pics+[pic[a:a+res,b:b+res] for a in range(0,self.nblocks*res) for b in range(0,self.nblocks*res)]
            xstep=pic.shape[0]//self.nx
            ystep=pic.shape[1]//self.ny
            def std(pic):
                return jnp.mean(geometry.postmap(jnp.std)(pic))
            for a in range(0,pic.shape[0],xstep):
                for b in range(0,pic.shape[1],ystep):
                    block=pic[a:a+xstep,b:b+ystep]
                    block=jax.image.resize(block,(res,res,3),jax.image.ResizeMethod.LANCZOS3)
                    if std(block)>std(pic)/2:
                        pics.append(block)
            
        self.pics=jnp.stack(pics)
        
    def getnext(self,n):
        pics=rnd.choice(nextkey(),self.pics,(n,))
        pics=jax.image.resize(pics,pics.shape[:-3]+(res,res,3),jax.image.ResizeMethod.LANCZOS3)
        return pics

def browse(outpath='weights',filter='session_'):
    filenames=[f for f in os.listdir(outpath) if filter in f]
    filenames.sort(key=lambda name:int(name.split('_')[-1]))
    print('options:')
    for s in filenames: print(s)
    counter=int(input('\nInput number:\n{}'.format(filter)))
    return os.path.join(outpath,'{}{}'.format(filter,counter))

def getpath(reload: bool,outpath='weights'):
    prevsessions=[f for f in os.listdir(outpath) if 'session_' in f]
    prevsessions.sort(key=lambda name:int(name.split('_')[1]))

    if reload:
        print('options:')
        for s in prevsessions: print(s)
        counter=int(input('\nInput session number:\nsession_'))
        return os.path.join(outpath,'session_{}'.format(counter))
    else:
        counter=max([0]+[int(f.split('_')[1]) for f in prevsessions])+1
        outpath=os.path.join(outpath,'session_{}'.format(counter))
        os.makedirs(outpath,exist_ok=True)
        return outpath
    
if __name__=='__main__':
    print(jax.devices())
    ndevs=jax.device_count()
    pmap=(ndevs>1)
    t0=time.time()
    t_00=time.time()
    
    loader=Loader(args.img_path_train)
    batchsize=int(cfg['batchsize'])

    checknans=cfg['checknans']=='True'
    ignorenans=cfg['ignorenans']=='True'
    print('checknans: {}'.format(checknans))
    print('ignorenans: {}'.format(ignorenans))

    pics0=loader.getnext(batchsize)
    bw=False
    if bw:
        pics0=pics0[...,:1]

    backend=jax.lib.xla_bridge.get_backend()
    
    outpath=getpath(reload=reload)

    if reload:
        path=outpath
        tp_path=browse(path,filter='trainer_and_params_')
        trainer,initialparams,losshist=util.load(tp_path)
        istart=len(losshist)
    else:
        istart=0

        painter=artist.Combo(
            mod.ConvNet(widesize=5,resnetlayers=1,orders=2,features=32,kernelsize=(3,3),ConvTranspose=False,normtype='pixel'),
            [
            artist.SolidBackground(),
            artist.Brush(nstrokes=50,width=10,nsteps=5,ndots=5,sampleseparate=True),
            artist.Brush(nstrokes=50,width=3,nsteps=5,ndots=5,sampleseparate=True),
            artist.Brush(nstrokes=50,width=5,nsteps=1,ndots=2,stepsize=2,sampleseparate=True),
            artist.Brush(nstrokes=50,width=5,nsteps=1,ndots=2,stepsize=2,sampleseparate=True),
            ]
        )
        reconstructor=mod.ConvNet(widesize=5,resnetlayers=10,orders=2,features=3,ConvTranspose=True)
        trainer=Trainer(
            painter,
            reconstructor,
            realism=.5
        )

        initialparams=trainer.init(nextkey(),pics0)
        util.save(dict(painter=painter,reconstructors=reconstructor,trainer=trainer),outpath,'models')
        losshist=[]

    fn=lambda params,pics:trainer.apply(
        params,
        pics,
        )

    valgrad=jax.jit(jax.value_and_grad(fn,has_aux=True))


    state=TrainState.create(apply_fn=None,params=initialparams,tx=optax.adamw(.001,weight_decay=.001))

    recentstates=deque([])
    recentparams=deque([])
    recentgrads=deque([])
    recentinputs=deque([])
    recent_aux=deque([])

    nantonum=jax.jit(lambda t:tree_map(lambda a:jnp.nan_to_num(a),t))
    clipgrad=jax.jit(lambda grad: util.clipgrad(grad,1.0))


    for i in range(istart,100001):
        try:

            ###########################################################################

            pics=loader.getnext(batchsize)

            if bw:
                pics=jnp.mean(pics,axis=-1)[...,None]

            (loss,aux),grads=valgrad(state.params,pics)

            losses={k:jnp.mean(v) for k,(v,w) in aux['losses'].items()}
            #lm=aux['loss_magnitudes']

            flatgrads,treedef=tree_flatten(grads)
            anynans=[jnp.any(jnp.isnan(a)) for a in flatgrads]
            gradnans=jnp.any(jnp.array(anynans))
            lossnans=jnp.isnan(loss)
            nans=(lossnans or gradnans)
            if nans:
                print('loss is nan?: {}'.format(lossnans))
                print('grad is nan?: {}'.format(gradnans))

            grads=clipgrad(grads)

            ###########################################################################

            recentstates.appendleft(state)
            recentparams.appendleft(state.params)
            recentinputs.appendleft(pics)
            recentgrads.appendleft(grads)
            recent_aux.appendleft(aux)
            if i>5:
                recentstates.pop()
                recentparams.pop()
                recentgrads.pop()
                recentinputs.pop()
                recent_aux.pop()

            dt=(time.time()-t0); t0=time.time()
            if i in [100,250,500] or i%1000==0:
                util.save(state.params,outpath,'params_{}'.format(i))
                util.save((trainer,state.params,losshist),outpath,'trainer_and_params_{}'.format(i))

                fig,axs=plt.subplots(1,3,figsize=(12,4))

                for k,D in enumerate([pics,aux['paintings'],aux['recs']]):
                    util.imshow(D[0],vmin=0,vmax=1,ax=axs[k])

                plt.savefig(os.path.join(outpath,'{}.jpg'.format(i)))
                plt.close('all')

                
            if i%10000==0:
                util.save(aux,outpath,'aux_{}'.format(i))

            if i%250==0:
                util.save(state.params,outpath,'params')
                util.save(losshist,outpath,'metrics')

                nd=len(aux['displayable'])
                fig,axs=plt.subplots(1,nd,figsize=(4*nd,4))

                for i,D in enumerate(aux['displayable']):
                    if isinstance(D,dict):
                        [(mode,D)]=D.items()
                        if mode=='quiver':
                            axs[i].quiver(D[0,::-5,::5,1],-D[0,::-5,::5,0])
                    else:
                        util.imshow(D[0],vmin=0,vmax=1,ax=axs[i])

                plt.savefig(os.path.join(outpath,'example.pdf'))
                plt.close('all')


            if nans: # or i==10:
                util.save(recentparams,outpath,'inspect_nan_params')
                util.save(recentgrads,outpath,'inspect_nan_grads')
                util.save(recentinputs,outpath,'inspect_nan_inputs')
                util.save(recent_aux,outpath,'inspect_nan_aux')
                print(tree_unflatten(treedef,anynans))

                print('restoring from recentstates')
                state=recentstates[-1]

            else:
                state=state.apply_gradients(grads=grads)
                print(losses)
                losshist.append(losses)


            if i%50==1:
                metrics=losshist
                fig,axs=plt.subplots(2)
                ylim=0
                for k in metrics[0].keys():
                    metric=jnp.array([slice[k] for slice in metrics])
                    axs[0].plot(metric,label=k)
                    axs[0].legend()
                    ylim=max(ylim,2*jnp.quantile(metric,.9))
                #axs[0].semilogy()
                axs[0].set_ylim(0,ylim)
                plt.savefig(os.path.join(outpath,'metrics.pdf'))
                plt.close('all')

                print()
                print('batch size {}'.format(batchsize))
                print('{:.3f} s per iteration'.format(dt))
                print()
                print('{:,.1f} pixels/s'.format(res*res*batchsize/dt))
                print()

            ###########################################################################


            ###########################################################################

            if i%10==0:
                #print([jnp.prod(jnp.array(b.shape)) for b in backend.live_buffers()])
                print(100*'#')
                print(len([a for a in backend.live_buffers() if jnp.prod(jnp.array(jnp.shape(a)))>100]))
                print(100*'#')

            if i==1:
                print(100*'#')
                print('compilation time {}'.format(time.time()-t_00))
                print(100*'#')

        except KeyboardInterrupt:
            quit()
            breakpoint()
