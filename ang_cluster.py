import twopoint
import numpy as np
import coord_trans as ct
from astropy.io import fits
from log_bins import log_bins
import pdb
import matplotlib.pyplot as plt
from matplotlib import ticker

def ang_cluster(data,rand,bins,
                data2=None,rand2=None,
                estimator='landy-szalay',
                error=None,
                outfile=None,plot=False,bincents=None,
                threads=4):

    """
    Wrapper for clustering-tree angular clustering code, (hopefully) 
    simplifying use. Also performs conversion of astronomical coordinates
    to x,y,z.

    :param data:
        The data for the measurement - can be either a string name of a
        fits table, or the data already read in. Needs columns 'RA', 'Dec'.
        Also needs column 'reg' if doing a jackknife or field to field
        error estimate

    :param rand:
        Same as above, but for the random catalog

    :param bins
        The edges of the bins for the clustering estimate, in degrees.

    :param data2: (optional)
        If doing a cross-correlation, supply a second data set

    :param rand2: (optional)
        If doing a cross-correlation, supply a second random set. If 
        data2 is set but not rand2, will use same randoms for both.

    :param estimator: (optional)
        Estimator used by clustering-tree. 
        Options are: 'landy-szalay', 'hamilton', or 'standard'
        Defaults to landy-szalay

    :param error: (optional)
        Options are 'jackknife', 'field-to-field', or 'poisson'. For the first
        two, will use 'reg' column to split samples and do error estimate

    :param outfile: (optional)
        Set to a string file name to write results out to a file
  
    :param plot: (optional)
        If not None or False will make a plot of the results.  If set to 
        a string file name, will write plot to a file, with the type set
        by the filename extension.

    :param bincents: (optional)
        The centers of the bins to use for plotting. If not set, will take the
        mean between successive bin edges.
    
    :param threads: (optional)
        Number of CPU threads used by clustering-tree. Defaults to 4.
    """
    
    #MAD If input data is a filename, read it in
    if type(data) is str:
        data=fits.open(data)[1]

    #MAD Put data on unit sphere (r=1), convert to x,y,z
    data_r=np.ones(len(data.data))
    dx,dy,dz=ct.sph2cart(data_r, data.data['ra'], data.data['dec'])
    dataxyz=np.array([dx,dy,dz]).transpose()
    
    #MAD If input randoms is a filename, read it in
    if type(rand) is str:
        rand=fits.open(rand)[1]

    #MAD Put randoms on unit sphere, convert to x,y,z
    rand_r=np.ones(len(rand.data))
    rx,ry,rz=ct.sph2cart(rand_r, rand.data['ra'], rand.data['dec'])   
    randxyz=np.array([rx,ry,rz]).transpose()

    #MAD If error hasn't been set, construct tree without fields
    if not error:
        error=None
        dtree = twopoint.clustering.tree(dataxyz)
        rtree = twopoint.clustering.tree(randxyz)

    #MAD If error is set, use "reg" column of input to construct tree with fields
    if error:
        if 'REG' in data.data.columns.names:
            d_fields=np.array(data.data['reg'])
            r_fields=np.array(rand.data['reg'])
            dtree = twopoint.clustering.tree(dataxyz, d_fields.astype('int'))
            rtree = twopoint.clustering.tree(randxyz, r_fields.astype('int'))

    #MAD Run clustering-tree autocorrelation code
    results = twopoint.angular.autocorr(dtree, rtree, bins,
                                        est_type=estimator,
                                        err_type=error, num_threads=threads)

    #MAD If an outfile is supplied, write results to file
    if outfile:
        f=open(outfile, 'w')
        f.write(str(results))
        f.close()

    #MAD If plot keyword is set, make a plot
    if plot:
        #MAD If bin centers not supplied, just take the mean of consecutive binedges
        if type(bincents) is not list:
            bincents=np.zeros(len(bins)-1)
            for i in range(0,len(bins)-2):
                bincents[i]=np.mean([bins[i],bins[i+1]])
            
        fig = plt.figure(figsize=(8,5))
        ax=fig.add_subplot(1,1,1)
        pltcorr, = ax.plot(bincents,results.estimate(),'o',color='black',markersize=5)
        pltline, = ax.plot(bincents,results.estimate(),'--',color='grey')
        if error:
            ax.errorbar(bincents,results.estimate(),yerr=results.error(),fmt='none',ecolor='black', capsize=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True,linestyle='--')
        plt.tick_params(axis='both',which='both',labelsize=12,direction='in',labelleft=True,labelbottom=True,top=True,right=True)    
        plt.xlabel(r'$\theta\ [deg]$', fontsize=12)
        plt.ylabel(r'$\omega_{\theta}$', fontsize=14)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        #MAD If the plot keyword is string, assume it's a file name and write plot to file
        if type(plot) is str:
            outtype=plot.split('.')
            plt.savefig(plot,format=outtype[-1])

        #MAD if plot is set but not a filename, just show the plot
        if plot is True:
            plt.show()
        
    print(results)

    return results

if __name__ == '__main__': 
    print 'You should really run this as a module...'
