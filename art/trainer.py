from art import artist
import flax.linen as nn
import jax.numpy as jnp
from art import util
from typing import List
from art.models import CosLoss
from . import geometry
from .geometry import *
import matplotlib.pyplot as plt
import jax


class Trainer(nn.Module):
    painter: nn.Module
    reconstructor: nn.Module
    figurative:float=1
    realism:float=.1

    def __call__(self,pics,training=True,realism=None,figurative=None,logbalance=False,**kw):
        pca_pics,pca_info=PCA(pics)
        _pics_=jnp.concatenate([pics,pca_pics],axis=-1)

        paintings,aux=self.painter(_pics_,**kw)
        recs=self.reconstructor(jax.image.resize(paintings,pics.shape,jax.image.ResizeMethod.LANCZOS3))

        interior=interiormask(pics[...,0])
        dist=lambda diff: jnp.mean(jnp.abs(interior[...,None]*diff))

        if realism is None:
            realism=self.realism
        if figurative is None:
            figurative=self.figurative

        losses=dict()
        if training:
            losses['realism']=(dist(paintings-pics),realism)
            losses['figurative']=(dist(recs-pics),figurative)

        aux_trainer=dict(
            paintings=paintings,recs=recs,losses=losses,pca_pics=pca_pics,pca_info=pca_info,
            )
        aux_trainer.update(aux)
        aux_trainer['displayable']=[pics]+aux['displayable']+[paintings,recs]

        if logbalance:
            loss=sum([w*jnp.log(l) for l,w in losses.values()])
        else:
            loss=sum([w*l for l,w in losses.values()])
        return loss,aux_trainer
    

def PCA(X):
    mean=jnp.mean(X,axis=(-3,-2))
    X=X-mean[...,None,None,:]
    cov=jnp.mean(X[...,:,None]*X[...,None,:],axis=(-4,-3))
    _,basis=jnp.linalg.eigh(cov)
    Y=jnp.sum(X[...,None]*basis[...,None,None,:,:],axis=-2)
    return Y,(mean,basis)

def unPCA(Y,pca_info):
    mean,basis=pca_info
    X=jnp.sum(Y[...,None,:]*basis[...,None,None,:,:],axis=-1)
    X=X+mean[...,None,None,:]
    return X
