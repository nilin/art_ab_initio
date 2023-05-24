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
from .artist import *



class Brush2(Element):
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
    def __call__(self,latents,key=None,upscaling=1,**kw):

        if key is None:
            key=rnd.PRNGKey(0)

        k1,k2,k3,k4,*_=rnd.split(key,10)

        potentials=nn.Dense(features=self.nfeatures)(latents)
        lr,V,Vi,density=LocalFrame(self.framescale)(potentials)

        interior=geometry.interiormask(jnp.ones_like(latents[...,0]),.1)
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
        #rtransform=postmap(smooth_nn)(rtransform)
        rtransform=jnp.reshape(rtransform,rtransform.shape[:-1]+(2,2))

        rtransform=(rtransform-postmap(postmap(mean))(rtransform))*(1-self.c0)+jnp.array([[1,0],[0,0]])*self.c0
        rtransform=rtransform*interior[...,None,None]

        rtransform_bias=nn.Dense(features=2)(potentials)
        #rtransform_bias=postmap(smooth_nn)(rtransform_bias)
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
        shape=ends.shape[:-2]+(ends.shape[-2]*upscaling,)*2
        segmentinfo=sqdist_to_segment(XYstack[:-1],XYstack[1:],shape)

        width=self.getfields(latents,'widths')[None,...]*self.width
        width=upscale(width,upscaling)*upscaling

        softstack0=1/(1+segmentinfo['sqdist']/width**2)
        strokewidth=SegmentFilter(applyfilter=False)(latents,softstack0)[...,None,None]
        hardness=SegmentFilter(applyfilter=False)(latents,softstack0)[...,None,None]
        lenwise_opacity=SegmentFilter(applyfilter=False)(latents,softstack0)
        opacity=divide(lenwise_opacity,jnp.max(lenwise_opacity,axis=0)[None])[...,None,None]
        #hardness=0.5+hardness*0.5
        width=width*strokewidth

        softstack=1/(1+segmentinfo['sqdist']/width**2)
        segmentstack=util.straight_thru(softstack,(segmentinfo['sqdist']<width**2)*(hardness+(1-hardness)*softstack))
        segmentstack=segmentstack*opacity

        badsegments=(jnp.mean(segmentstack,axis=(-2,-1))>.01)
        badsegments=badsegments[...,None,None]
        softstack=softstack*(1-badsegments)
        segmentstack=segmentstack*(1-badsegments)
        #segmentstack=jnp.nan_to_num(segmentstack)
        #segmentstack=SegmentFilter()(latents,segmentstack,hardness=5)
        layers=jnp.max(segmentstack,axis=0)

        if self.sampleseparate:
            densities=upscale(densities,upscaling)
            layers=util.straight_thru(layers*densities,layers)
        else:
            density=upscale(density,upscaling)
            layers=util.straight_thru(layers*density[None],layers)

        layers=divide(layers,jnp.max(layers))*self.alpha
        #layers=util.straight_thru(1,layers>.02)*layers

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
    

class HardCoded(Element):
    nsteps: int=10
    ndots: int=25
    width:float=1
    stepsize: float=10
    blendmode:str='pushaside'
    nstrokes:int=25
    nfeatures:int=3
    framescale:int=4
    color: Tuple[float,float,float]=None
    alpha:float=1

    @nn.compact
    def __call__(self,x,key=None,upscaling=1,**kw):

        if key is None:
            key=rnd.PRNGKey(0)

        k1,k2,k3,k4,*_=rnd.split(key,10)

        interior=geometry.interiormask(jnp.ones_like(x[...,0]),.1)

        lr,V,Vi,density=LocalFrame(self.framescale)(x)
        tr=lr[...,0,0]+lr[...,1,1]
        I=tr[...,None,None]*jnp.eye(2)
        P=I-lr
        #starts=choose(k1,density*interior,self.nstrokes)
        starts=choose(k1,interior,self.nstrokes)

        #dirfield=rotate(grad(x))

        dirs=None
        dirstack=[]
        nodes=[starts]
        dirs=rnd.normal(k2,starts.shape[:-2]+(2,))

        for s in range(self.nsteps):
            P_=read_matrix(P,nodes[-1])
            dirs=mvprod(P_,dirs)
            lens=sqrt(jnp.sum(dirs**2,axis=-1))[...,None]
            dirs=divide(dirs,(lens+jnp.mean(lens)*.5))
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

        #width=self.getfields(latents,'widths')[None,...]*self.width
        width=self.width

        #upscaling
        shape=ends.shape[:-2]+(ends.shape[-2]*upscaling,)*2
        #width=upscale(width,upscaling)*upscaling

        segmentinfo=sqdist_to_segment(XYstack[:-1],XYstack[1:],shape)
        #softstack=1/(1+segmentinfo['sqdist']/width**2)
        #segmentstack=util.straight_thru(softstack,segmentinfo['sqdist']<width**2)
        segmentstack=(segmentinfo['sqdist']<width**2)

        badsegments=(jnp.mean(segmentstack,axis=(-2,-1))>.01)
        badsegments=badsegments[...,None,None]
        #softstack=softstack*(1-badsegments)
        segmentstack=segmentstack*(1-badsegments)
        #segmentstack=(jnp.arange(len(segmentstack))*segmentstack.T).T

        #segmentstack=SegmentFilter()(latents,segmentstack)
        layers=jnp.max(segmentstack,axis=0)

        #densities=upscale(densities,upscaling)
        #layers=util.straight_thru(layers*densities,layers)

        layers=divide(layers,jnp.max(layers))*self.alpha

        #colorlayers,colors=ColorFilter(directread=True)(x[...,:3],layers)
        #colorlayers,colors=ColorFilter(directread=True)(x[...,:3],layers)
        colors=read_vector(x,starts)
        colors=colors+(1-colors)*rnd.uniform(k2,colors.shape[:-1])[...,None]*.02
        colorlayers=jnp.ones_like(layers[...,None])*colors[...,None,None,:]
        colorlayers=jnp.concatenate([layers[...,None],colorlayers],axis=-1)

        aux=dict(
                segments=segmentstack,
                alphalayers=layers,
                colorlayers=colorlayers,
                displayable=[]
                )
        return colorlayers,aux