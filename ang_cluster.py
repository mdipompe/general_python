import twopoint
import numpy as np
from astropy.io import fits
import pdb
import matplotlib.pyplot as plt
from matplotlib import ticker
import pidly
import os
import pdb

'''
Convert spherical to xyz coordinates
'''
def sph2cart(r, phi, theta, degree=False):
    if degree:
        phi = phi * (np.pi/180.)
        theta = theta * (np.pi/180.)
    rcos_theta = r * np.cos(theta)
    x = rcos_theta * np.cos(phi)
    y = rcos_theta * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def ang_cluster(data,rand,bins,
                data2=None,rand2=None,
                estimator='landy-szalay',
                error=None,
                outfile=None,plot=False,bincents=None,
                threads=4,
                n=10,frac=0.05):

    """
    Wrapper for clustering-tree angular clustering code, (hopefully) 
    simplifying use. Also performs conversion of astronomical coordinates
    to x,y,z, as well as slitting data into regions for error estimation.

    :param data:
        The data for the measurement - can be either a string name of a
        fits table, or the data already read in. Needs keys 'RA', 'Dec'.
        Also needs key 'reg' if doing a jackknife or field to field
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

    :param n: (optional)
        If data need to be split by regions for error estimation, n is the number
        of regions per side for the splitting (default n=10)

    :param frac: (optional)
        If data need to be split by regions for error estimation, frac is the fraction
        of the random sample to use to calculate splits (default frac=0.05)

    """
    
    #MAD If input data is a filename, read it in
    if type(data) is str:
        data=fits.open(data)[1]
        data=data.data
        
    #MAD Put data on unit sphere (r=1), convert to x,y,z
    data_r=np.ones(len(data))
    dx,dy,dz=sph2cart(data_r, data['ra'], data['dec'], degree=True)
    dataxyz=np.array([dx,dy,dz]).transpose()
    
    #MAD If input randoms is a filename, read it in
    if type(rand) is str:
        rand=fits.open(rand)[1]
        rand=rand.data

    #MAD Put randoms on unit sphere, convert to x,y,z
    rand_r=np.ones(len(rand))
    rx,ry,rz=sph2cart(rand_r, rand['ra'], rand['dec'], degree=True)   
    randxyz=np.array([rx,ry,rz]).transpose()

    #MAD If error hasn't been set, construct tree without fields
    if not error:
        error=None
        dtree = twopoint.clustering.tree(dataxyz)
        rtree = twopoint.clustering.tree(randxyz)

    #MAD If error is set, use "reg" column of input to construct tree with fields
    #MAD IF reg column doesn't exist, call IDL code to make them
    if error:
        if (not 'REG' in data.names) or (not 'REG' in rand.names):
            c1=fits.Column(name='RA', array=data['RA'], format='D')
            c2=fits.Column(name='DEC', array=data['DEC'], format='D')
            tmp1=fits.BinTableHDU.from_columns([c1,c2])
            tmp1.writeto('data_tmp.fits')
            c1=fits.Column(name='RA', array=rand['RA'], format='D')
            c2=fits.Column(name='DEC', array=rand['DEC'], format='D')
            tmp2=fits.BinTableHDU.from_columns([c1,c2])
            tmp2.writeto('rand_tmp.fits')
            idl = pidly.IDL()
            idl.pro('split_regions_gen','data_tmp.fits','rand_tmp.fits',
                    n=n,frac=frac,
                    data_fileout='data_reg.fits',rand_fileout='rand_reg.fits',
                    figures=True,split_file="jack_splits.txt",countoff=True)
            os.remove('data_tmp.fits')
            os.remove('rand_tmp.fits')
            
            data=fits.open('data_reg.fits')[1]
            data=data.data
            rand=fits.open('rand_reg.fits')[1]
            rand=rand.data
            
        d_fields=np.array(data['reg'])
        r_fields=np.array(rand['reg'])
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
#        pltline, = ax.plot(bincents,results.estimate(),'--',color='grey')
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
