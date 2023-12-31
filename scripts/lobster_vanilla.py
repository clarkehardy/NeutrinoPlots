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
    parser.add_argument('-npoints',type=int,default=200)
    parser.add_argument('-nsamples',type=float,default=1e4)
    parser.add_argument('-outfile',type=str,default='')
    args = parser.parse_args()
    npoints = args.npoints
    nsamples = args.nsamples
    outfile = args.outfile
    if outfile=='':
        outfile = 'lobster_vanilla.png'

    print('Loading the oscillation parameters...')
    params = load_params()

    print('Making the plot...')
    fig,axs = vanilla(params=params,npoints=npoints,nsamples=nsamples)
    fig.savefig(outfile)
    print('Figure saved to {}'.format(outfile))
