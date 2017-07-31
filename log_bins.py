'''
Define bins uniformly spaced in log space. Uses numpys logspace, but allows
user to specify number of points per dex, and returns bin edges as well as 
centers
'''
import numpy as np
import matplotlib.pyplot as plt

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

