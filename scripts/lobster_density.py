import argparse
import numpy as np
import matplotlib.pyplot as plt
# my default plot style (not available unless manually installed)
try:
    plt.style.use('clarke-default')
except:
    pass
from lobster.plotting import *
from utils.load_data import *


if __name__=='__main__':
    """Make the lobster density plot.
    """

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path_no')
    parser.add_argument('path_io')
    parser.add_argument('--nbins',type=int,default=200)
    parser.add_argument('--npoints',type=int,default=100000)
    parser.add_argument('--outfile',type=str,default='')
    parser.add_argument('--data-path',type=str,default='')
    parser.add_argument('--allowed',action='store_true',default=False)
    parser.add_argument('--kde',action='store_true',default=False)
    parser.add_argument('--sum',action='store_true',default=False)
    args = parser.parse_args()
    path_no = args.path_no
    path_io = args.path_io
    nbins = args.nbins
    npoints = args.npoints
    outfile = args.outfile
    data_path = args.data_path
    allowed = args.allowed
    kde = args.kde
    sum = args.sum
    if outfile=='':
        outfile = 'lobster_density.png'
    if data_path == '':
        data_path = None
    style = 'kde' if kde else 'hist'

    # load the mcmc sample chain
    print('Loading the MCMC sample chains...')
    samples_no = np.loadtxt(path_no)
    samples_io = np.loadtxt(path_io)
    if allowed:
        params = load_params()
    else:
        params = None

    print('Making the plot...')
    fig,axs = density(samples_no=samples_no, samples_io=samples_io, nbins=nbins, npoints=npoints,\
                      params=params, style=style, sum=sum, cmap='magma_r', data_save_path=data_path)
    fig.savefig(outfile)
    print('Figure saved to {}'.format(outfile))
