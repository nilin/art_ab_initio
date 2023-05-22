# implemented by nilin

from dataclasses import dataclass
import jax.numpy as jnp
import jax
import flax.linen as nn
import jax.random as rnd
from typing import Tuple, Dict, Any, List
import numpy as np
from functools import partial
from jax.lax import stop_gradient as sg
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax.flatten_util import ravel_pytree as fullyflatten
from art import util
from . import geometry
from .geometry import *




class NN(nn.Module):
    def norm(self,x):
        if self.normtype=='instance':
            return util.InstanceNorm()(x)
        if self.normtype=='pixel':
            x=x-postmap(jnp.mean)(x)
            x=divide(x,sqrt(jnp.sum(x**2,axis=-1))[...,None])
            return x
        if self.normtype is None:
            return x



class ConvNet(NN):
    widesize: int=7
    f32: int=32
    orders: int=2
    resnetlayers: int=6
    downsample: bool=False
    upsample: bool=False
    features: int=None
    normtype: str='instance'
    kernelsize: Tuple[int,int]=(3,3)
    ConvTranspose: bool=True
    fmax=64

    @nn.compact
    def __call__(self,x):

        ws,f32=self.widesize,self.f32

        if not self.upsample:
            x=nn.Conv(features=f32,kernel_size=(ws,ws))(x)
            x=self.norm(x)
            x=nn.relu(x)

            for i in np.arange(self.orders)+1:
                nf=min(f32*2**i,self.fmax)
                x=nn.Conv(features=nf,kernel_size=self.kernelsize,strides=(2,2))(x)
                x=self.norm(x)
                x=nn.relu(x)
        
        for i in range(self.resnetlayers):
            x=ResNetLayer(kernel_size=self.kernelsize)(x)
            
        if not self.downsample:
            for i in range(self.orders):
                nf=min(f32*2**(self.orders-i),self.fmax)

                if self.ConvTranspose:
                    x=nn.ConvTranspose(\
                        features=nf,
                        strides=(2,2),
                        kernel_size=self.kernelsize)(x)
                else:
                    x=self.double(x)
                    x=nn.Conv(\
                        features=nf,
                        kernel_size=self.kernelsize)(x)

                x=self.norm(x)
                x=nn.relu(x)

            x=nn.Conv(features=x.shape[-1] if self.features is None else self.features,kernel_size=(ws,ws))(x)
        return x

    def double(self,x):
        *batchshape,m,n,d=x.shape
        return jax.image.resize(x,(*batchshape,2*m,2*n,d),jax.image.ResizeMethod.LINEAR)

class ResNetLayer(NN):
    normtype: str='pixel'
    kernel_size: tuple[int,int]=3

    @nn.compact
    def __call__(self,x):
        features=x.shape[-1]
        y=x
        y=nn.Conv(features=features,kernel_size=self.kernel_size)(y) 
        y=self.norm(y)
        y=nn.relu(y)
        y=nn.Conv(features=features,kernel_size=self.kernel_size)(y) 
        y=self.norm(y)
        return x+y




class TrivialLoss(nn.Module):
    def __call__(self,*k):
        return dict()


class CosLoss(nn.Module):
    blocksize:int=4

    def __call__(self,x,y,return_nearest=False):
        shape=x.shape
        x=self.combine(x)
        y=self.combine(y)
        x=self.normalize(x)
        y=self.normalize(y)

        y=jnp.reshape(y,(-1,y.shape[-1]))

        piclosses=jnp.min(1-jnp.inner(x,y),axis=-1)
        out=dict(cosloss=(jnp.mean(piclosses),1))

        I=(jnp.argmax(jnp.inner(x,y),axis=-1)[...,None]-jnp.arange(y.shape[0]))==0

        new_x=jnp.dot(I,y)
        new_x=jnp.reshape(new_x,new_x.shape[:-1]+(self.blocksize,self.blocksize,-1))
        new_x=jnp.swapaxes(new_x,-4,-3)
        new_x=jnp.reshape(new_x,shape)
        return out,dict(new_x=new_x)


    def normalize(self,x):
        x=x-jnp.mean(x,axis=-1)[...,None]
        x=x/(jnp.linalg.norm(x,axis=-1)[...,None]+.00001)
        return x

    #def combine(self,x):
    #    I=jnp.arange(0,x.shape[-2],self.blocksize)

    #    out=[]
    #    for i in range(self.blocksize):
    #        for j in range(self.blocksize):
    #            out.append(x[...,I+i,:,:][...,I+j,:])

    #    #def loop(ij,out):
    #    #    i=ij//self.blocksize
    #    #    j=ij%self.blocksize
    #    #    grid=x[...,I+i,:,:][...,I+j,:]
    #    #    out[ij]=grid
    #    #out=jax.lax.fori_loop(0,self.blocksize**2,loop,[None]*self.blocksize**2)

    #    return jnp.stack(out,axis=-1)

    def combine(self,x,**kw):
        x=self.block(x,self.blocksize,self.blocksize,**kw)
        x=jnp.reshape(x,x.shape[:-3]+(-1,))
        return x

    @staticmethod
    def block(x,a,b,stack=False):
        w,h,d=x.shape[-3:]
        x=jnp.reshape(x,x.shape[:-3]+(w//a,a,h//b,b,d))
        x=jnp.moveaxis(x,-4,-3)
        if stack:
            x=jnp.reshape(x,(-1,)+x.shape[-3:])
        return x








class ColorNet(nn.Module):
    endsize: int=2
    features: int=32

    @nn.compact
    def __call__(self,x):
        x=util.resize_img(x,self.endsize)
        x=nn.Dense(features=self.features)(x)
        x=nn.relu(x)
        x=nn.Dense(features=self.features)(x)
        return x





def compare(interpreter,fixedparams,X1,X2):
    Y1,intermediates_1=interpreter(jax.lax.stop_gradient(fixedparams),X1,capture=True)
    Y2,intermediates_2=interpreter(jax.lax.stop_gradient(fixedparams),X2,capture=True)
    out=0
    intermediates_1,_=tree_flatten(intermediates_1)
    intermediates_2,_=tree_flatten(intermediates_2)
    for I1,I2 in zip(intermediates_1,intermediates_2):
        out+=jnp.mean(jnp.abs(I1-I2))
    return out, Y1, Y2


class LengthScales(nn.Module):
    elements: List[ConvNet]
    features: int
    base: int=2

    @nn.compact
    def __call__(self,x):
        out=[]
        for k,e in enumerate(self.elements):
            y=self.downsample(x,self.base**k)
            y=e(y)
            y=self.upsample(y,self.base**k)
            out.append(y)
        out=jnp.stack(out,axis=-1)
        return nn.Dense(features=1)(out)[...,0]

    def reshape_2d(self,X,res):
        *batchshape,_,_,d=X.shape
        return jax.image.resize(X,(*batchshape,res,res,d),jax.image.ResizeMethod.LANCZOS3)
    
    def upsample(self,X,ratio):
        res=X.shape[-2]*ratio
        return self.reshape_2d(X,res)

    def downsample(self,X,ratio):
        res=X.shape[-2]//ratio
        return self.reshape_2d(X,res)
    

class BW(nn.Module):
    @nn.compact
    def __call__(self,x):
        x=jnp.mean(x,axis=-1)
        x=x-jnp.mean(x,axis=(-2,-1))[...,None,None]
        x=x/(jnp.std(x,axis=(-2,-1))[...,None,None]+.000001)
        return x[...,None]
    



class ParallelChannels(ConvNet):

    @nn.compact
    def __call__(self,x):
        x=jnp.moveaxis(x,-1,0)[...,None]
        x=super().__call__(x)
        x=x[...,0]
        x=jnp.moveaxis(x,0,-1)
        return x




    