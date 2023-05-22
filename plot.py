import matplotlib.pyplot as plt
import jax.numpy as jnp



def squeeze(L):
    if not isinstance(L,list):
        return L
    if len(L)==1:
        return squeeze(L[0])
    
    return [squeeze(l) for l in L]
    


def plot(pics,outpath=None):

    pics=squeeze(pics)

    depth=0
    test=pics
    while isinstance(test,list):
        test=test[0]
        depth+=1
    
    if depth==0:
        plt.imshow(pics)

    if depth==1:
        w=len(pics)
        fig,axs=plt.subplots(1,w,figsize=(4*w,4))
        for i in range(w):
            pic=pics[i]
            if pic.shape[-1]==4: pic=jnp.concatenate([pic[:,:,1:],pic[:,:,:1]],axis=-1)
            axs[i].imshow(pic)

    if depth==2:
        w,h=len(pics),len(pics[0])
        fig,axs=plt.subplots(w,h,figsize=(4*h,4*w))
        for i in range(w):
            for j in range(h):
                pic=pics[i][j]
                if pic.shape[-1]==4: pic=jnp.concatenate([pic[:,:,1:],pic[:,:,:1]],axis=-1)
                axs[i][j].imshow(pic)

    if outpath is None:
        plt.show()
    else:
        fig.savefig(outpath)
    plt.close('all')

#
#
#    if isinstance(pics,list):
#
#        if len(pics)==1:
#            pics=pics[0]
#
#        if isinstance(pics[0],list):
#            w,h=len(pics),len(pics[0])
#            fig,axs=plt.subplots(w,h,figsize=(4*h,4*w))
#            for i in range(w):
#                for j in range(h):
#                    pic=pics[i][j]
#                    if pic.shape[-1]==4: pic=jnp.concatenate([pic[:,:,1:],pic[:,:,:1]],axis=-1)
#                    axs[i][j].imshow(pic)
#        else:
#            w=len(pics)
#            fig,axs=plt.subplots(1,w,figsize=(12,8))
#            for i in range(w):
#                pic=pics[i]
#                if pic.shape[-1]==4: pic=jnp.concatenate([pic[:,:,1:],pic[:,:,:1]],axis=-1)
#                axs[i].imshow(pic)
#    else:
#        pic=pics
#        if pic.shape[-1]==4: pic=jnp.concatenate([pic[:,:,1:],pic[:,:,:1]],axis=-1)
#        plt.imshow(pic)
#
#
#
#
#
#
#