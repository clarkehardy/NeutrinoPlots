import argparse
import numpy as np
import matplotlib.pyplot as plt
# my default plot style (not available unless manually installed)
try:
    plt.style.use('clarke-default')
except:
    pass
from nuplots.nuplots import *
from nuplots.load_data import *


if __name__=='__main__':
    """Make the vanilla lobster plot.
    """

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--npoints',type=int,default=200)
    parser.add_argument('--nsamples',type=float,default=1e4)
    parser.add_argument('--outfile',type=str,default='')
    args = parser.parse_args()
    npoints = args.npoints
    nsamples = args.nsamples
    outfile = args.outfile
    if outfile == '':
        outfile = 'lobster_vanilla.png'

    print('Loading the oscillation parameters...')
    params = load_params()

    print('Making the plot...')
    fig,axs = lobster_vanilla(params=params, npoints=npoints, nsamples=nsamples)
    fig.savefig(outfile)
    print('Figure saved to {}'.format(outfile))
