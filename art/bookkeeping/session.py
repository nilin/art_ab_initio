import jax.random as rnd
import jax.numpy as jnp

rootkey=rnd.PRNGKey(0)

def nextkey(n=None):
    global rootkey

    if n==None:
        rootkey,sendkey=rnd.split(rootkey)
        return sendkey
    else:
        keys=rnd.split(rootkey,n+1)
        rootkey,sendkeys=keys[0], keys[1:]
        return sendkeys

import time
starttime=time.ctime().replace(' ','_')