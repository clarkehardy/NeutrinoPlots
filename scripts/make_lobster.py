import argparse
import numpy as np
from plot_results import *


if __name__=='__main__':
    '''
    Make the lobster plot.
    '''

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path_no')
    parser.add_argument('path_io')
    parser.add_argument('-nbins',type=int,default=200)
    parser.add_argument('-npoints',type=int,default=100000)
    parser.add_argument('-outfile',type=str,default='')
    args = parser.parse_args()
    path_no = args.path_no
    path_io = args.path_io
    nbins = args.nbins
    npoints = args.npoints
    outfile = args.outfile
    if outfile=='':
        outfile = 'lobster.png'

    # load the mcmc sample chain
    print('Loading the MCMC sample chains...')
    samples_no = np.loadtxt(path_no)
    samples_io = np.loadtxt(path_io)
    params = load_params()

    print('Making the plot...')
    fig,axs = lobster_plot(samples_no=samples_no,samples_io=samples_io,nbins=nbins,npoints=npoints,params=params)
    fig.savefig(outfile)
    print('Figure saved to {}'.format(outfile))

