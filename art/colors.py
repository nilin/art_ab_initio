import jax.numpy as jnp
from typing import Tuple


def sum_alphas(A,**kw):
    return 1-jnp.product(1-A,**kw)

def add_alphas(*layers):
    return sum_alphas(jnp.stack(layers,axis=0),axis=0)

def add_alpha_colors(top,bottom):
    top_alpha,top_color=top
    bottom_alpha,bottom_color=bottom
    alpha_channel=add_alphas(top_alpha,bottom_alpha)
    top_alpha_rel=top_alpha/(alpha_channel+.0001)
    color_channel=top_alpha_rel[...,None]*top_color+(1-top_alpha_rel)[...,None]*bottom_color
    return alpha_channel,color_channel

def smudge(mask,alpha,colors):
    alpha_=neighbors(alpha[...,None])
    colors_=neighbors(colors)
    newalpha=jnp.mean(alpha_,axis=-1)
    newcolors=jnp.mean(alpha_*colors_,axis=-1)/(alpha+.0001)
    mask=mask[...,None]
    return mask*newalpha+(1-mask)*alpha, mask*newcolors+(1-mask)*colors

def neighbors(A):
    I=jnp.arange(A.shape[-3])
    K=jnp.arange(A.shape[-2])
    return jnp.stack([A[...,I-1,:,:],A[...,I+1,:,:],A[...,K-1,:],A[...,K+1,:]],axis=-1)

def combine(layer,A,smudgemasks=tuple([])):
    alpha,colors=A[...,0],A[...,1:]
    alphalayer,colorlayer=layer[...,0],layer[...,1:]
    for mask in smudgemasks:
        alpha,colors=smudge(mask,alpha,colors)
    alpha,colors=add_alpha_colors((alphalayer,colorlayer),(alpha,colors))
    return jnp.concatenate([alpha[...,None],colors],axis=-1)

def combine_flat(layer,A):
    return fixate(combine(layer,opaque(A)))

def merge(layers,fixate=False):
    for i,layer in enumerate(layers):
        if isinstance(layer,tuple):
            layer,smudgemasks=layer
        else:
            smudgemasks=()
        if i==0:
            out=layer
        else:
            out=combine(layer,out,smudgemasks)
    if fixate:
        return fixate_fn(out)
    else:
        return out
    
def merge_unordered(layers,axis=0):
    axis=axis%len(layers.shape)
    if isinstance(layers,list):
        layers=jnp.stack(layers,axis=axis)
    colors,alphas=layers[...,1:],layers[...,0]
    alpha=sum_alphas(alphas,axis=axis)
    color=jnp.mean(colors*alphas[...,None],axis=axis)/(jnp.mean(alphas[...,None],axis=axis)+.0001)
    return zip_alpha(alpha,color)

def combinecolorlayers(colors,alphas,axis):
    if axis<0: axis-=1 # don't include color dimension in axis input above
    return jnp.mean(colors*alphas[...,None],axis=axis)/(jnp.mean(alphas[...,None],axis=axis)+.0001)

def fixate(layer):
    assert layer.shape[-1]==4
    return combine(layer,jnp.ones_like(layer))[...,1:]

fixate_fn=fixate

def opaque(A):
    if A.shape[-1]==4: return A
    return jnp.concatenate([jnp.ones(A.shape[:-1]+(1,)),A],axis=-1)

def zip_alpha(alpha,colors):
    return jnp.concatenate([alpha[...,None],colors],axis=-1)

def unzip_alpha(alphacolors):
    return alphacolors[...,0],alphacolors[...,1:]

def apply_unzipped(f,layer):
    return zip_alpha(*f(*unzip_alpha(layer)))