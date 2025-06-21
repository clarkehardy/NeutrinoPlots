import emcee
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool,cpu_count
from utils.load_data import *
from utils.util_funcs import *
from lobster.prob_funcs import *
from lobster.plotting import *


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
    np.seterr(invalid='ignore')

    # load the chi-squared data from which the likelihood functions will be constructed
    chi2_m2_beta_func = load_endpoint_data()
    chi2_halflife_func = load_0vbb_data()
    chi2_osc_funcs = load_osc_data(inverted=inverted)

    # construct arguments to be passed to the log-probability function
    chi2_funcs = [chi2_m2_beta_func] + [chi2_halflife_func] + list(chi2_osc_funcs)
    params = load_params()

    # use mean as initial guesses for the parameters
    sigma_mean = 0.16
    delta_m2_21_mean = params['delta_m2_21'][0]
    delta_m2_23_mean = params['delta_m2_23'][0]*2.*(0.5-inverted)
    theta_12_mean = params['theta_12'][0]*np.pi/180.
    theta_13_mean = params['theta_13'][0]*np.pi/180.
    alpha_21_mean = np.pi
    delta_minus_alpha31_mean = np.pi

    # use standard deviation to constrain initial guesses
    sigma_err = 0.16/3.
    delta_m2_21_err = params['delta_m2_21'][1]
    delta_m2_23_err = params['delta_m2_23'][1]
    theta_12_err = params['theta_12'][1]*np.pi/180.
    theta_13_err = params['theta_13'][1]*np.pi/180.
    alpha_21_err = np.pi/3.
    delta_minus_alpha31_err = np.pi/3.

    # build initial vectors for the mcmc
    initial_mean = np.array([sigma_mean,delta_m2_21_mean,delta_m2_23_mean,\
                             theta_12_mean,theta_13_mean,alpha_21_mean,delta_minus_alpha31_mean])
    initial_err = np.array([sigma_err,delta_m2_21_err,delta_m2_23_err,\
                            theta_12_err,theta_13_err,alpha_21_err,delta_minus_alpha31_err])
    ndim = len(initial_mean)
    p0 = [np.random.normal(loc=initial_mean,scale=initial_err) for i in range(nwalkers)]

    # make sure we're not requesting more cores than there are available
    if ncores>cpu_count():
        print('Warning: {} cores requested but only {} available!'.format(ncores,cpu_count()))
        ncores = cpu_count()

    print('Running MCMC using {} cores'.format(ncores))
    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, \
                                        moves=emcee.moves.StretchMove(a=20),args=[inverted,chi2_funcs,params])
        
        print("Running burn-in...")
        p0,_,_ = sampler.run_mcmc(p0,1000,progress=True)
        sampler.reset()

        print("Running production...")
        pos,prob,state = sampler.run_mcmc(p0,niter,progress=True)

    print('Saving the results...')
    samples = sampler.flatchain
    np.savetxt(filename,samples)

    print('Results saved to {}.'.format(filename))

    if plot:
        if inverted:
            samples_no = None
            samples_io = samples
        else:
            samples_no = samples
            samples_io = None

        fig,_ = density(samples_no=samples_no,samples_io=samples_io,params=params)
        fig.savefig(outdir+'lobster.png')

        print('Plot saved to {}'.format(outdir+'lobster.png'))