import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

'''
Define bins uniformly spaced in log space. Uses numpys logspace, but allows
user to specify number of points per dex, and returns bin edges as well as 
centers
'''
def log_bins(low=0.001, high=2.1, nperdex=5):
    num = ((np.log10(high) - np.log10(low)) * nperdex) + 1
    bin_edges = np.logspace(np.log10(low), np.log10(high), num=num)
    step = np.log10(bin_edges[1]) - np.log10(bin_edges[0])

    bin_cents = np.zeros(len(bin_edges) - 1)
    bin_cents[0] = np.log10(low) + (step/2)
    for i in range(1,len(bin_cents)):
        bin_cents[i] = bin_cents[i-1] + step
    bin_cents = 10. ** bin_cents

    return bin_edges, bin_cents

'''
Generate random indices with no repeats for random sub-sampling
'''
def rand_indices(inlen,outlen,seed=None):
    if outlen > inlen:
        print('rand_indices error: '
              'N_out > N_in, cannot generate non-repeating random indices')
        return
    if seed:
        np.random.seed(seed)
    indx = np.arange(0,inlen)
    np.random.shuffle(indx)
    indx = indx[0:outlen]
    return indx

'''
Read in fits file and get just the data portion
'''
def readfits(infile,ext=1):
    data=fits.open(infile)[ext]
    return data.data

    
