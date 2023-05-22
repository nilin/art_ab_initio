import jax.numpy as jnp
import jax.random as rnd
import numpy as np
from jax.tree_util import tree_map, tree_unflatten, tree_flatten
from typing import Dict
import flax.linen as nn
import jax
import jax.scipy.signal as jsi
import jax.scipy.stats as jst
import pickle
import os
from functools import partial


nothing=10**(-6)

def mean_but_keepdims(x,*keepdims):
    ndims=len(x.shape)
    dims=tuple([i for i in range(ndims) if (i not in keepdims and i-ndims not in keepdims)])
    return jnp.mean(x,axis=dims)

def center_along_last(x):
    mean=mean_but_keepdims(x,-1)
    return x-mean[None,...,:]

def straight_thru(x,y):
    return x-jax.lax.stop_gradient(x)+jax.lax.stop_gradient(y)

def sync(A,dim):
    k=A.shape[dim]
    return jnp.repeat(jnp.expand_dims(jnp.mean(A,axis=dim),axis=dim),k,axis=dim)

def softtrunc01(x):
    return nn.sigmoid(x-1/2)

def softtrunc(x,ab):
    a,b=ab
    x_=(x-a)/(b-a)
    return (b-a)*softtrunc01(x_)+a

def hardtrunc(x,a,b):
    return jnp.minimum(jnp.maximum(x,a),b)

def hardsigmoid(x):
    return hardtrunc(x+1/2,0,1)

def shift(x,a,b,sigmoid=hardsigmoid):
    return (b-a)*sigmoid(x)+a

def clipgrad(A,t=1):
    return tree_map(lambda x:jnp.tanh(x/t)*t, A)

def trunc(X,a,b):
    return jnp.minimum(jnp.maximum(X,a),b)

def broadcast(A,shape):
    newshape=A.shape+(1,)*(len(shape)-len(A.shape))
    return jnp.reshape(A,newshape)
 
def nparams(T):
    t,D=tree_flatten(T)
    dims=0
    instances=1
    for x in t:
        I,b=x.shape
        dims+=b
        assert(I==instances or I==1 or instances==1)
        instances=max(instances,I)
    return instances,dims

def shapes(T):
    return tree_map(lambda A:A.shape,T)

def printshapes(T):
    print(shapes(T))

def cast(A,T):
    arrays,tdef=tree_flatten(T)
    s=0
    out=[]
    for a_ in arrays:
        I,b=a_.shape
        a=A[...,s:s+b]
        if I==1:
            a=jnp.mean(a,axis=-2)[...,None,:]
        out.append(a)
        s+=b
    return tree_unflatten(tdef,out)

def getstats(modeltree):
    arrays,tdef=tree_flatten(modeltree)
    return [(jnp.mean(a_),jnp.std(a_)) for a_ in arrays]

def printshapes(T,indent=0):
    if 'params' in T.keys():
        T=T['params']
    for k,A in T.items():
        if not isinstance(A,Dict):
            print('\n'*indent+'{}: {}'.format(k,A.shape))
        else:
            print(k)
            printshapes(A,indent+4)

def rangecost(A,ab=(0,1)):
    if isinstance(ab,float):
        a=b=ab
    else:
        a,b=ab
    return jnp.mean(nn.relu(A-b)+nn.relu(a-A))

def listprod(L):
    out=1
    for x in L:
        out*=x
    return out

def threshold(A,t):
    return nn.relu(A-t)

def psplit(A,ndevs):
    n,*m=A.shape
    return jnp.reshape(A,(ndevs,n//ndevs,*m))

def downsample(A,s):
    for k in range(10):
        if 2**k==s:
            return A
        m,n=A.shape[-3:-1]
        A=(A[...,jnp.arange(0,m,2),:,:]+A[...,jnp.arange(1,m,2),:,:])/2
        A=(A[...,jnp.arange(0,n,2),:]+A[...,jnp.arange(1,n,2),:])/2
    raise ValueError('s not power of 2')


@jax.custom_jvp
def prioritize(x,weight):
    return x

@prioritize.defjvp
def prioritize_jvp(x_,dx_):
    x,weight=x_
    dx,dw=dx_
    return x,weight*dx


import matplotlib.pyplot as plt
def show(A):
    plt.imshow(A)
    plt.show()

def imshow(img,ax=None,**kwargs):
    img=jnp.squeeze(img)
    if ax is None:
        plt.imshow(img,**kwargs)
    else:
        ax.imshow(img,**kwargs)



#InstanceNorm=lambda **kw: nn.GroupNorm(num_groups=None,group_size=1,**kw)


class PixelNorm(nn.Module):
    @nn.compact
    def __call__(self,x):
        norms=jnp.linalg.norm(x,axis=-1)+nothing
        return x/norms[...,None]
    
class InstanceNorm(nn.Module):
    @nn.compact
    def __call__(self,x):
        norms=jnp.sqrt(jnp.mean(x**2,axis=(-3,-2,-1)))+nothing
        return x/norms[...,None,None,None]

class FlatImageNorm(nn.Module):
    @nn.compact
    def __call__(self,x):
        norms=jnp.linalg.norm(x,axis=(-2,-1))
        return x/norms[...,None,None]

#class InstanceNorm(nn.Module):
#    @nn.compact
#    def __call__(self,x):
#        norms=jnp.sqrt(jnp.mean(jnp.sum(x**2,axis=-1),axis=(-2,-1)))
#        return x/norms[...,None,None,None]

#def nparams(T):
#    t,D=tree_flatten(T)
#    dims=0
#    for x in t:
#        dim_x=1
#        for d in x.shape:
#            dim_x*=d
#        dims+=dim_x
#    return dims
#    
#def cast(A,T):
#    t,D=tree_flatten(T)
#    s=0
#    out=[]
#    for a_ in t:
#        blocksize=np.product(np.array(a_.shape))
#        a=A[...,s:s+blocksize]
#        a=jnp.reshape(a,A.shape[:-1]+a_.shape)
#        out.append(a)
#        s+=blocksize
#    return tree_unflatten(D,out)

def flattenlast(A,k):
    s1,s2=A.shape[:-k],A.shape[-k:]
    d2=np.product(np.array(s2))
    return jnp.reshape(A,s1+(d2,))

def normalize(x):
    x=x-jnp.mean(x)
    x=x/jnp.std(x)
    return x

def save(data,*path):
    path=os.path.join(*path)
    with open(path,'wb') as f:
        pickle.dump(data,f)

def load(*path):
    path=os.path.join(*path)
    f=open(path,'rb')
    return pickle.load(f)



def away_from(x,c,ab):
    a,b=ab
    return ((x-c)/(b-c))*(x>c)+((c-x)/(c-a))*(x<c)


def resize_img(img,resolution=None,upratio=None,downratio=None):
    *batchshape,oldres,_,d=img.shape
    if upratio is not None:
        resolution=oldres*upratio
    if downratio is not None:
        resolution=oldres//downratio
    if oldres==resolution:
        return img
    else:
        return jax.image.resize(img,(*batchshape,resolution,resolution,d),jax.image.ResizeMethod.LANCZOS3)

def resize(img,resolution):
    *batchshape,oldres,_=img.shape
    if oldres==resolution:
        return img
    else:
        return jax.image.resize(img,(*batchshape,resolution,resolution),jax.image.ResizeMethod.LANCZOS3)

def smooth2d(img,width=1):
    x=jnp.arange(-2*width,2*width+1)
    normal=jnp.exp(-(x/width)**2/2)
    kernel=normal[:,None]*normal[None,:]
    kernel=kernel/jnp.sum(kernel)
    return jsi.convolve(img,kernel,mode='same')

def smooth_img(img,width=1):
    f=jax.vmap(partial(smooth2d,width=width),in_axes=-1,out_axes=-1)
    if len(img.shape)==3:
        return f(img)
    else:
        return jax.vmap(f)(img)
    

def search(tree,key):
    if key is None:
        return None
    if isinstance(tree,dict):
        if key in tree:
            return tree[key]
        else:
            return search([v for v in tree.values()],key)
    if isinstance(tree,list):
        for b in tree:
            hit=search(b,key)
            if hit is not None:
                return hit
    return None


def randkey(pic,n=1):
    key=rnd.PRNGKey(jnp.int(jnp.round(100000*jnp.mean(pic))))
    if n==1:
        return key
    else:
        return rnd.split(key,n)