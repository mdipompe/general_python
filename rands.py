import numpy as np

def rand_indices(inlen,outlen,seed=None):
    if seed:
        np.random.seed(seed)
    indx = np.arange(0,inlen)
    np.random.shuffle(indx)
    indx = indx[0:outlen]
    return indx
