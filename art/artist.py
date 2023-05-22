import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
from art import colors
import jax.random as rnd
from jax.image import ResizeMethod as rm
import flax.linen as nn
from functools import partial
from art import util, models
from .geometry import *
from . import geometry as geometry






class Element(nn.Module):
    pass

class Brush(Element):
    nsteps: int=10
    ndots: int=25
    width:float=1
    stepsize: float=10
    blendmode:str='pushaside'
    nstrokes:int=25
    nfeatures:int=3
    framescale:int=1
    color: Tuple[float,float,float]=None
    sampleseparate:bool=True
    alpha:float=1
    c0:float=1

    @nn.compact
    def __call__(self,latents,key=None,render=False,upscaling=1,border=.1):

        if key is None:
            key=rnd.PRNGKey(0)

        k1,k2,k3,k4,*_=rnd.split(key,10)

        potentials=nn.Dense(features=self.nfeatures)(latents)
        lr,V,Vi,density=LocalFrame(self.framescale)(potentials)

        interior=geometry.interiormask(jnp.ones_like(latents[...,0]),border)
        mean=partial(maskedmean,interior)

        if self.sampleseparate:
            densities=nn.Dense(features=self.nstrokes)(latents)
            densities=nn.softmax(densities,axis=(-3,-2))
            #densities=ScalarPotentials(self.nstrokes)(latents)

            densities=jnp.moveaxis(densities,-1,0)
            starts=choose(k1,densities*interior)
            density=jnp.mean(densities,axis=0)
        else:
            starts=choose(k1,density*interior,self.nstrokes)

        rtransform=nn.Dense(features=4)(potentials)
        rtransform=postmap(smooth_nn)(rtransform)
        rtransform=jnp.reshape(rtransform,rtransform.shape[:-1]+(2,2))

        rtransform=(rtransform-postmap(postmap(mean))(rtransform))*(1-self.c0)+jnp.array([[1,0],[0,0]])*self.c0
        rtransform=rtransform*interior[...,None,None]

        rtransform_bias=nn.Dense(features=2)(potentials)
        rtransform_bias=postmap(smooth_nn)(rtransform_bias)
        rtransform_bias=rtransform_bias*interior[...,None]

        transform=mmprod(V,rtransform,Vi)

        dirs=None
        dirstack=[]
        nodes=[starts]
        dirs=read_vector(rtransform_bias,starts)

        for s in range(self.nsteps):
            transform_=read_matrix(transform,nodes[-1])
            dirs=mvprod(transform_,dirs)

            dirs=dirs/jnp.sqrt(jnp.sum(dirs**2,axis=-1)+1)[...,None]*self.stepsize
            ends=move(nodes[-1],dirs)
            dirstack.append(dirs)
            nodes.append(ends)

        XY=geometry.XY(starts)
        dirstack=jnp.stack(dirstack)
        nodes=jnp.stack(nodes)

        XYstack=XY[None,...]+jnp.cumsum(dirstack,axis=0)
        XYstack=jnp.concatenate([XY[None,...],XYstack],axis=0)
        XYstack=jax.image.resize(XYstack,(self.ndots,)+XYstack.shape[1:],jax.image.ResizeMethod.CUBIC)
        XYstack=XYstack*upscaling

        width=self.getfields(latents,'widths')[None,...]*self.width

        #upscaling
        shape=ends.shape[:-2]+(ends.shape[-2]*upscaling,)*2
        width=upscale(width,upscaling)*upscaling

        segmentinfo=sqdist_to_segment(XYstack[:-1],XYstack[1:],shape)
        softstack=1/(1+segmentinfo['sqdist']/width**2)
        segmentstack=util.straight_thru(softstack,(segmentinfo['sqdist']<width**2)*softstack)

        badsegments=(jnp.mean(segmentstack,axis=(-2,-1))>.01)
        badsegments=badsegments[...,None,None]
        softstack=softstack*(1-badsegments)
        segmentstack=segmentstack*(1-badsegments)
        segmentstack=jnp.nan_to_num(segmentstack)

        segmentstack=SegmentFilter()(latents,segmentstack)
        layers=jnp.max(segmentstack,axis=0)

        if self.sampleseparate:
            densities=upscale(densities,upscaling)
            layers=util.straight_thru(layers*densities,layers)
        else:
            density=upscale(density,upscaling)
            layers=util.straight_thru(layers*density[None],layers)

        layers=divide(layers,jnp.max(layers))*self.alpha

        if self.color is None:
            colorlayers,colors=ColorFilter()(latents,layers)
        else:
            colorlayers=jnp.concatenate([layers[...,None],jnp.ones_like(layers[...,None])*jnp.array(self.color)],axis=-1)

        aux=dict(
                localrotation=lr,
                segments=segmentstack,
                alphalayers=layers,
                colorlayers=colorlayers,
                displayable=[density,{'quiver':transform[...,0]},potentials[...,:3]]
                )
        return colorlayers,aux

    def getfields(self,latents,name):
        interior=geometry.interiormask(jnp.ones_like(latents[...,0]),.1)
        widths=nn.Dense(features=self.nstrokes,name=name)(latents)
        widths=jnp.moveaxis(widths,-1,0)
        widths=widths-maskedmean(widths,interior[None,...])
        widths=widths/maskedstd(widths,interior[None,...])
        widths=nn.sigmoid(widths)
        return widths
    
def upscale(img,ratio):
    *batch,n,_=img.shape
    return jax.image.resize(img,tuple(batch)+(n*ratio,)*2,jax.image.ResizeMethod.CUBIC)

def rescale(img,N):
    return jax.image.resize(img,img.shape[:-2]+(N,N),jax.image.ResizeMethod.CUBIC)


class SolidBackground(Element):
    @nn.compact
    def __call__(self,latents,upscaling=1,**kw):
        colorproposals=nn.Dense(features=3)(latents)
        color=jnp.mean(colorproposals,axis=(-3,-2))
        color=jnp.concatenate([jnp.ones_like(color[...,:1]),color],axis=-1)
        shape=latents.shape[:-3]+(latents.shape[-2]*upscaling,)*2
        colorlayers=jnp.ones(shape+(1,))*color[...,None,None,:]
        colorlayers=colorlayers[None]
        aux=dict(displayable=[])
        return colorlayers,aux


class Combo(nn.Module):
    conv_enc: models.ConvNet
    elements: List[nn.Module]
    alphas: List[float]=None
    recolor:bool=True
    colors: List[Tuple[float,float,float]]=None
    threshold=.05

    @nn.compact
    def __call__(self,x,upscaling=1,repeats=1,fixate=True,layerweights=None,bg=None,**kw):

        if bg is None:
            bg=jnp.ones(x.shape[:-3]+2*(x.shape[-2]*upscaling,)+(3,))

        bg=colors.opaque(bg)

        colorlayers=bg[None,...]
        displayable=[]
        aux=dict()

        if layerweights is None:
            layerweights=[True]*100

        for i,(e,m) in enumerate(zip(self.elements,layerweights)):

            if not m:
                continue

            for _ in range(repeats):
                bg=jax.lax.stop_gradient(colors.merge(colorlayers,fixate=True))
                bg=jax.image.resize(bg,x.shape[:-1]+(bg.shape[-1],),jax.image.ResizeMethod.CUBIC)
                xbg=jnp.concatenate([x,bg],axis=-1)

                latents=self.conv_enc(xbg)
                newcolorlayers,aux_=e(latents,upscaling=upscaling,**kw)
                colorlayers=jnp.concatenate([colorlayers,newcolorlayers])
                aux[str(i)]=aux_
                displayable+=aux_['displayable']

        aux['displayable']=displayable
        return colors.merge(colorlayers,fixate=fixate),aux


class Filter(nn.Module):
    @staticmethod
    def upscale(latents,layers):
        latents=postmap(partial(rescale,N=layers.shape[-2]))(latents)
        return latents

class ColorFilter(Filter):
    ncolors:int=3
    @nn.compact
    def __call__(self,latents,layers):
        oglayers=layers
        interior=geometry.interiormask(layers,.1)
        layers=layers*interior
        latents=self.upscale(latents,layers)
        locallatents=maskedmean(latents[None,...],layers[...,None],axis=(-3,-2))
        colors=nn.Dense(features=self.ncolors)(locallatents)
        colors=colors-jnp.median(colors)
        colors=nn.sigmoid(colors)
        #colorlayers=jnp.concatenate([layers[...,None],jnp.ones_like(layers[...,None])*colors[...,None,None,:]],axis=-1)
        colorlayers=jnp.concatenate([oglayers[...,None],jnp.ones_like(layers[...,None])*colors[...,None,None,:]],axis=-1)
        return colorlayers,colors

class SegmentFilter(Filter):
    @nn.compact
    def __call__(self,latents,segments):
        ogsegments=segments
        interior=geometry.interiormask(segments,.1)
        segments=segments*interior
        latents=self.upscale(latents,segments)
        locallatents=maskedmean(latents[None,None,...],segments[...,None],axis=(-3,-2))
        confs=nn.Dense(features=1)(locallatents)[...,0]
        confs=confs-jnp.min(confs)
        confs=confs-jnp.quantile(confs,.5)
        confs=nn.sigmoid(2*confs)*.99+.01

        return confs[...,None,None]*ogsegments

class LocalFrame(nn.Module):
    framescale:int=2

    @nn.compact
    def __call__(self,latents):
        lr=jnp.mean(postmap(partial(getlocalrotation,scale=self.framescale))(latents),axis=-1)
        #eigs,V=jax.lax.stop_gradient(jnp.linalg.eigh(lr))
        eigs,V=jnp.linalg.eigh(lr)
        Vi=jnp.swapaxes(V,-2,-1)
        density=eigs[...,-1]
        return lr,V,Vi,density
