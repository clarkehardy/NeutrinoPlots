import argparse
import numpy as np
from lobster.plotting import *
from utils.load_data import *


if __name__=='__main__':
    '''
    Make the lobster density plot.
    '''

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path_no')
    parser.add_argument('path_io')
    parser.add_argument('-nbins',type=int,default=200)
    parser.add_argument('-npoints',type=int,default=100000)
    parser.add_argument('-outfile',type=str,default='')
    parser.add_argument('-allowed',action='store_true',default=False)
    args = parser.parse_args()
    path_no = args.path_no
    path_io = args.path_io
    nbins = args.nbins
    npoints = args.npoints
    outfile = args.outfile
    allowed = args.allowed
    if outfile=='':
        outfile = 'lobster_density.png'

    # load the mcmc sample chain
    print('Loading the MCMC sample chains...')
    samples_no = np.loadtxt(path_no)
    samples_io = np.loadtxt(path_io)
    if allowed:
        params = load_params()
    else:
        params = None

    print('Making the plot...')
    fig,axs = density(samples_no=samples_no,samples_io=samples_io,nbins=nbins,npoints=npoints,params=params)
    fig.savefig(outfile)
    print('Figure saved to {}'.format(outfile))
