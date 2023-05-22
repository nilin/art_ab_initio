import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.scipy.signal as jsi
from functools import partial
from jax.image import ResizeMethod as rm

no=nothing=.001

def transverse_pos(in_res,out_res,potential):
    slope=jax.lax.stop_gradient(slope(potential,ratio=out_res/in_res)+nothing)
    return divide(potential,slope)
    
def grad_single(h,ratio=1):
    x_kernel=jnp.array([[1],[-1]])
    y_kernel=jnp.array([[1,-1]])
    dx=jsi.convolve(h,x_kernel,mode='same')
    dy=jsi.convolve(h,y_kernel,mode='same')
    out=jnp.stack([dx,dy],axis=-1)*ratio
    return out

def grad(h,ratio=1):
    return wmap(grad_single,h,dims=2,ratio=ratio)

def wmap(f,x,dims=2,**kwargs):
    batchshape=x.shape[:-dims]
    x=x.reshape((-1,)+x.shape[-dims:])
    y=jax.vmap(partial(f,**kwargs))(x)
    y=y.reshape(batchshape+y.shape[1:])
    return y
    
def convolve_nn(x,kernel):
    return wmap(jsi.convolve,x,dims=2,in2=kernel,mode='same')
    
def convolve_nnd(x,kernel):
    return jax.vmap(convolve_nn,in_axes=(-1,None),out_axes=-1)(x,kernel)

def slope(y,ratio=1,square=False):
    grad_=grad(y,ratio)
    squareslope=jnp.sum(grad_**2,axis=-1)
    if square:
        return squareslope
    else:
        return sqrt(squareslope)

def Hessian(h,ratio=1):
    grad=grad(h,ratio)
    dx=grad[...,0]
    dy=grad[...,1]
    ddx=grad(dx,ratio)
    ddy=grad(dy,ratio)
    return jnp.stack([ddx,ddy],axis=-1)
    
def eigs(H):
    m=(H[...,0,0]+H[...,1,1])/2
    D=H[...,0,0]*H[...,1,1]-H[...,0,1]*H[...,1,0]
    eigvals=(m-jnp.sqrt(m**2-D+nothing),m+jnp.sqrt(m**2-D+nothing))
    l1,l2=eigvals

    a=H[...,0,0]
    bc=H[...,0,1]
    d=H[...,1,1]
    out=jnp.array([[l1-d,bc],[bc,l2-a]])
    out=jnp.moveaxis(out,0,-1)
    eigvecs=jnp.moveaxis(out,0,-1)
    eigvecs=jnp.swapaxes(eigvecs,-2,-1)
    eigvecs=divide(eigvecs,norm(eigvecs,axis=-1)[...,None])
    return eigvals,eigvecs

def A_to_P(A):
    (l1,l2),eigvecs=eigs(A)
    P=A-l1[...,None,None]*jnp.eye(2)
    return P,l2-l1

def interiormask(alphalayers,t=.1):
    m=alphalayers.shape[-2]
    I=jnp.arange(m)/m
    I=(I>t)*(I<1-t)
    return I[:,None]*I[None,:]
        
def decay(dist,width,squared=False):
    reldist=2*jnp.abs(dist)/width
    if squared: reldist=reldist**2
    return 1/(1+reldist)

def smooth_nn(x,x_scale=1):
    if x_scale>1:
        N=x.shape[-2]
        n=N//x_scale
        x=jax.image.resize(x,x.shape[:-2]+(n,n),rm.LANCZOS3)
        k=smoothingkernel(1)
        x=convolve_nn(x,k)
        x=jax.image.resize(x,x.shape[:-2]+(N,N),rm.LANCZOS3)
    k=smoothingkernel(1)
    x=convolve_nn(x,k)
    return x

def smoothingkernel(width):
    x=jnp.arange(-2*width,2*width+1)
    normal=jnp.exp(-(x/width)**2/2)
    kernel=normal[:,None]*normal[None,:]
    return kernel/jnp.sum(kernel)

def transverse(vfield):
    P=vfield[...,:,None]*vfield[...,None,:]
    I=(P[...,0,0]+P[...,1,1])[...,None,None]*jnp.eye(2)
    return I-P

def sqdist_to_segment(starts,ends,shape):
    lenwise=wavefronttime(starts,ends,shape,relative=False)
    transverse=wavefronttime(starts,ends,shape,rotate=True,relative=False)
    length=jnp.sqrt(jnp.sum((1+(ends-starts)**2),axis=-1))[...,None,None]
    lendist=jnp.maximum(0,jnp.maximum(lenwise-length,-lenwise))
    sqdist=transverse**2+lendist**2
    return dict(transverse=transverse,lenwise=lenwise,lendist=lendist,length=length,sqdist=sqdist)


def wavefronttime(XY0,XY1,shape,rotate=False,relative=True):

    x=jnp.arange(shape[-2])[:,None]
    y=jnp.arange(shape[-1])[None,:]

    #x0=self.X(starts)[...,None,None]
    #y0=self.Y(starts)[...,None,None]

    x0=XY0[...,None,None,0]
    y0=XY0[...,None,None,1]
    x1=XY1[...,None,None,0]
    y1=XY1[...,None,None,1]

    dx=x1-x0
    dy=y1-y0

    if rotate:
        dx,dy=-dy,dx

    t=dx*(x-x0)+dy*(y-y0)

    if relative:
        t=divide(t,dx**2+dy**2)
    else:
        t=divide(t,sqrt(dx**2+dy**2))

    return t
#
#    def getsegment(self,starts,ends,shaperef):
#        forward=self.time(starts,ends,shaperef,relative=False)
#        transverse=self.time(starts,ends,shaperef,rotate=True,relative=False)
#        length=jnp.linalg.norm(ends-starts,axis=-1)[...,None,None]
#
#        lendist=jnp.maximum(0,jnp.maximum(forward-length,-forward))
#        sqdist=transverse**2+lendist**2
#        reldist2=sqdist/(self.width**2+no)
#        return util.straight_thru(1/(1+reldist2),self.stylize(transverse,lendist))

def read_matrix(mfield,onehot):
    return jax.vmap(read_vector,in_axes=(-1,None),out_axes=-1)(mfield,onehot)

def read_vector(vfield,onehot):
    return jax.vmap(read_scalar,in_axes=(-1,None),out_axes=-1)(vfield,onehot)

def read_scalar(field,onehot):
    return divide(jnp.sum(onehot*field,axis=(-2,-1)),jnp.sum(onehot,axis=(-2,-1)))
    #denom=jnp.sum(onehot,axis=(-2,-1))
    #mask=(denom>0)
    #denom=mask*denom+(1-mask)*1000
    #return jnp.sum(onehot*field,axis=(-2,-1))/denom

def read_covector(cvfield,onehot,dirs=None,dirsfield=None):
    covs=read_vector(cvfield,onehot)
    if dirs is None:
        dirs=read_vector(dirsfield,onehot)
    #dirs=dirs/jnp.sqrt(dirs[...,0:1]**2+dirs[...,1:2]**2+1)
    return jnp.sum(covs*dirs,axis=-1)

def move(starts,dirs):
    newxy=XY(starts)+jnp.round(dirs)
    newxy=jnp.minimum(jnp.maximum(newxy,0),starts.shape[-1]-1)
    return onehot(XY=newxy,shaperef=starts)
    
def controlsize(v,size):
    #norm=jnp.sqrt(jnp.sum(v**2,axis=-1)+jnp.mean(v**2)+no)[...,None]
    #mask=(norm>jnp.mean(norm))
    #return v/norm*size*mask
    return divide(v*size,sqrt(jnp.sum(v**2,axis=-1)+jnp.mean(v**2)+no)[...,None])
    #return v/jnp.sqrt(jnp.sum(v**2,axis=-1)+no)[...,None]*size

def balance(*xws):
    out=0
    for x,w in xws:
        out+=controlsize(x,w)
    return out

def choose_2d(key,p,n):
    p_=jnp.ravel(p)
    p_=divide(p_,jnp.sum(p_))
    a_=jnp.arange(len(p_))
    choices=rnd.choice(key,a_,p=p_,shape=(n,))
    a=jnp.reshape(a_,p.shape)
    onehot=(choices[:,None,None]==a[None,:,:])
    return onehot
    
def choose(key,p,n=1,flatten_if_n1=True):
    if len(p.shape)==2:
        return choose_2d(key,p,n)
    else:
        p_=jnp.reshape(p,(-1,)+p.shape[-2:])
        keys=rnd.split(key,p_.shape[0])
        choices=jax.vmap(partial(choose_2d,n=n),in_axes=(0,-3),out_axes=-3)(keys,p_)
        choices=jnp.reshape(choices,(p.shape if n==1 and flatten_if_n1 else (n,)+p.shape))
        return choices

def X(p):
    x=jnp.arange(p.shape[-2])
    p=jnp.sum(p,axis=-1)
    return divide(jnp.sum(p*x,axis=-1),jnp.sum(p,axis=-1))

def Y(p):
    y=jnp.arange(p.shape[-1])
    p=jnp.sum(p,axis=-2)
    return divide(jnp.sum(p*y,axis=-1),jnp.sum(p,axis=-1))

def XY(p):
    return jnp.stack([X(p),Y(p)],axis=-1)
    
def convdelta(f,p=None,xy=None,shaperef=None):
    if p is not None:
        x0=X(p)[...,None,None]
        y0=Y(p)[...,None,None]
        shaperef=p
    if xy is not None:
        x0=xy[...,0,None,None]
        y0=xy[...,1,None,None]

    x=jnp.arange(shaperef.shape[-2])[...,:,None]
    y=jnp.arange(shaperef.shape[-1])[...,None,:]
    dx=x-x0
    dy=y-y0
    return f(dx,dy)

def rotate(v):
    return jnp.stack([-v[...,1],v[...,0]],axis=-1)

def dist2(p0,p1):
    dx=X(p1)-X(p0)
    dy=Y(p1)-Y(p0)
    return jnp.sqrt(dx**2+dy**2)

def dist(p):
    x0=X(p)[...,None,None]
    y0=Y(p)[...,None,None]
    x=jnp.arange(p.shape[-2])[:,None]
    y=jnp.arange(p.shape[-1])[None,:]
    dx=x-x0
    dy=y-y0
    return jnp.sqrt(dx**2+dy**2)

def onehot(X=None,Y=None,XY=None,shaperef=None):
    if XY is not None:
        X=XY[...,0]
        Y=XY[...,1]
    x=jnp.arange(shaperef.shape[-2])[:,None]
    y=jnp.arange(shaperef.shape[-1])[None,:]
    return (x==X[...,None,None])*(y==Y[...,None,None])

def maskedmean(f,mask,**kw):
    return divide(jnp.mean(f*mask,**kw),jnp.mean(mask,**kw))

def maskedvar(f,mask,**kw):
    return maskedmean(f**2,mask,**kw)-maskedmean(f,mask,**kw)**2

def maskedstd(f,mask,**kw):
    return sqrt(maskedvar(f,mask,**kw))

def magnitude(x):
    return jnp.sqrt(jnp.mean(x**2+no))

def getmax(A):
    A_=jnp.reshape(A,A.shape[:-2]+(-1,))
    k=jnp.argmax(A_,axis=-1)
    onehot=(k[...,None]==jnp.arange(A_.shape[-1]))
    return jnp.reshape(onehot,A.shape)

def symmetrize(A):
    A_=jnp.swapaxes(A,-2,-1)
    return A/2+A_/2
    
def mmprod(A,B,*Cs):
    if len(Cs)==0:
        return jnp.sum(A[...,:,:,None]*B[...,None,:,:],axis=-2)
    else:
        AB=mmprod(A,B)
        return mmprod(AB,*Cs)
    
def mvprod(M,v):
    return jnp.sum(M*v[...,None,:],axis=-1)

def divide(x,y,defaultvalue=0):
    mask=(y!=0)
    y_=mask*y+(1-mask)*1
    out=x/y_
    out=mask*out+(1-mask)*defaultvalue
    return out

def sqrt(x):
    mask=(x>0)
    x_=mask*x+(1-mask)*1
    return mask*jnp.sqrt(x_)

def norm(x,**kw):
    return sqrt(jnp.sum(x**2,**kw))

postmap=lambda f:jax.vmap(f,in_axes=-1,out_axes=-1)

def getlocalrotation(f,scale):
    lr=grad(f)
    lr=lr[...,:,None]*lr[...,None,:]
    lr=postmap(postmap(partial(smooth_nn,x_scale=scale)))(lr)
    return lr
