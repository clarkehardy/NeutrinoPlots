import argparse
import numpy as np
from nuplots.load_data import *
from nuplots.lobster import *
from nuplots.nuplots import *


if __name__=='__main__':
    """Run the MCMC here.
    """

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--io',action='store_true',default=False)
    parser.add_argument('--ncores',type=int,default=10)
    parser.add_argument('--nwalkers',type=int,default=500)
    parser.add_argument('--niter',type=int,default=1000)
    parser.add_argument('--outdir',type=str,default='')
    parser.add_argument('--plot',action='store_true',default=False)
    args = parser.parse_args()
    inverted = args.io
    ncores = args.ncores
    nwalkers = args.nwalkers
    niter = args.niter
    outdir = args.outdir
    plot = args.plot
    filename = 'samples_{}_{}.npy'.format(['no','io'][inverted],nwalkers*niter)
    if outdir != '':
        outdir = outdir + '/'
        filename = outdir + filename

    samples = mcmc(inverted, ncores, nwalkers, niter, filename)

    if plot:
        if inverted:
            samples_no = None
            samples_io = samples
        else:
            samples_no = samples
            samples_io = None

        params = load_params()
        fig, _ = lobster_density(samples_no=samples_no, samples_io=samples_io, params=params)
        fig.savefig(outdir + 'lobster.png')

        print('Plot saved to {}'.format(outdir + 'lobster.png'))