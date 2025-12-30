from tqdm import tqdm
import emcee
from multiprocessing import Pool, cpu_count
from nuplots.load_data import *
from nuplots.mass_funcs import *

def get_3sigma_range(m_lightest, params, inverted=False, nsamples=1e4, sum=False):
    """For a single value of the lightest mass, find the 3 sigma range of
    the effective Majorana mass.
    """
    
    # get the allowed ranges from the params dictionary
    delta_m2_21_range = np.array(params['delta_m2_21'])[(2+2*inverted):(4 + 2*inverted)]
    delta_m2_23_range = np.array(params['delta_m2_23'])[(2+2*inverted):(4 + 2*inverted)]
    theta_12_range = np.array(params['theta_12'])[(2+2*inverted):(4 + 2*inverted)]*np.pi/180.
    theta_13_range = np.array(params['theta_13'])[(2+2*inverted):(4 + 2*inverted)]*np.pi/180.

    # sample values from within the 3 sigma allowed region
    nsamples = int(nsamples)
    delta_m2_21 = delta_m2_21_range[0] + np.random.beta(0.1, 0.1, nsamples)*(delta_m2_21_range[1] - delta_m2_21_range[0])
    delta_m2_23 = delta_m2_23_range[0] + np.random.beta(0.1, 0.1, nsamples)*(delta_m2_23_range[1] - delta_m2_23_range[0])
    theta_12 = theta_12_range[0] + np.random.beta(0.1, 0.1, nsamples)*(theta_12_range[1] - theta_12_range[0])
    theta_13 = theta_13_range[0] + np.random.beta(0.1, 0.1, nsamples)*(theta_13_range[1] - theta_13_range[0])
    alpha_21 = np.random.uniform(0, 2*np.pi, nsamples)
    delta_minus_alpha31 = np.random.uniform(0, 2*np.pi, nsamples)

    # initial limits to be replaced as more values are calculated
    m_bb_lower = 1e12
    m_bb_upper = 1e-12

    if sum:
        for i in range(nsamples):
            masses, success = basis_change(m_lightest, delta_m2_21[i], -delta_m2_23[i])
            if np.any(masses < 0) or not success:
                return np.nan, np.nan
            m_1, m_2, m_3 = masses
            m_betabeta = m_bb(m_1,m_2,m_3,np.sin(theta_12[i]),np.sin(theta_13[i]),\
                              np.cos(theta_12[i]),np.cos(theta_13[i]),alpha_21[i],delta_minus_alpha31[i])
            if m_betabeta > m_bb_upper:
                m_bb_upper = m_betabeta
            if m_betabeta < m_bb_lower:
                m_bb_lower = m_betabeta

    else:
        if inverted:
            for i in range(nsamples):
                # compute the effective Majorana mass for the inverted ordering
                m_1 = m1_io(m_lightest, delta_m2_23[i], delta_m2_21[i])
                m_2 = m2_io(m_lightest, delta_m2_23[i])
                m_betabeta = m_bb(m_1, m_2, m_lightest, np.sin(theta_12[i]), np.sin(theta_13[i]), \
                                  np.cos(theta_12[i]), np.cos(theta_13[i]), alpha_21[i], delta_minus_alpha31[i])
                
                # replace the bounds if the new values fall outside them
                if m_betabeta > m_bb_upper:
                    m_bb_upper = m_betabeta
                if m_betabeta < m_bb_lower:
                    m_bb_lower = m_betabeta

        else:
            for i in range(nsamples):
                # compute the effective Majorana mass for the normal ordering
                m_2 = m2_no(m_lightest, delta_m2_21[i])
                m_3 = m3_no(m_lightest, delta_m2_21[i], delta_m2_23[i])
                m_betabeta = m_bb(m_lightest, m_2, m_3, np.sin(theta_12[i]), np.sin(theta_13[i]), \
                                  np.cos(theta_12[i]), np.cos(theta_13[i]), alpha_21[i], delta_minus_alpha31[i])
                
                # replace the bounds if the new values fall outside them
                if m_betabeta > m_bb_upper:
                    m_bb_upper = m_betabeta
                if m_betabeta < m_bb_lower:
                    m_bb_lower = m_betabeta

    return m_bb_lower,m_bb_upper


def get_contours(params, inverted=False, npoints=100, nsamples=1e4, sum=False):
    """Get the upper and lower contours for the 3sigma allowed regions.
    """

    m_lightest = np.logspace(-5, 0, npoints)
    if sum:
        m_lightest = np.logspace(np.log10(5e-2), np.log10(2), npoints)
    m_lower = np.zeros_like(m_lightest)
    m_upper = np.zeros_like(m_lightest)

    order = ['normal','inverted']
    print('Computing the 3 sigma allowed region for {} points for {} ordering...'.format(npoints,order[inverted]))
    for i,m_light in enumerate(tqdm(m_lightest)):
        m_l,m_u = get_3sigma_range(m_light, params, inverted, nsamples, sum=sum)
        m_lower[i] = m_l
        m_upper[i] = m_u

    return m_lightest,m_lower,m_upper


def lnlike(theta, chi2_funcs, params, ln_prior):
    """Log-likehood function based on experimental data from oscillations and limits on the effective
    electron antineutrino mass and the effective Majorana mass.
    """

    # if the prior is negative infinity, don't bother computing the likelihoods
    if np.isneginf(ln_prior):
        return -np.inf

    # get the chi^2 functions to construct the likelihoods
    sigma,delta_m2_21,delta_m2_23,theta12,theta13,alpha21,alpha31_minus_delta = theta
    chi2_m2_beta,chi2_halflife,chi2_sin2_theta12,chi2_sin2_theta13,chi2_delta_m2_21,chi2_delta_m2_23 = chi2_funcs

    # constraints from oscillation data from nu-fit
    lnlike_sin2_theta12 = -chi2_sin2_theta12(np.sin(theta12)**2)/2.
    lnlike_sin2_theta13 = -chi2_sin2_theta13(np.sin(theta13)**2)/2.
    lnlike_delta_m21_2 = -chi2_delta_m2_21(delta_m2_21)/2.
    lnlike_delta_m32_2 = -chi2_delta_m2_23(delta_m2_23)/2.

    # constraints from KATRIN
    masses,success = basis_change(sigma,delta_m2_21,delta_m2_23)
    if success is False:
        return -np.inf
    m_beta = m_b(*masses,np.sin(theta12),np.sin(theta13),np.cos(theta12),np.cos(theta13))
    lnlike_m_beta = -chi2_m2_beta(m_beta**2)/2.

    # constraints from KamLAND-Zen
    m_betabeta = m_bb(*masses,np.sin(theta12),np.sin(theta13),np.cos(theta12),np.cos(theta13),alpha21,alpha31_minus_delta)
    halflife = T_half(m_betabeta,params)
    lnlike_halflife = -chi2_halflife(halflife)/2.

    result = lnlike_sin2_theta12 + lnlike_sin2_theta13 + lnlike_delta_m21_2 + lnlike_delta_m32_2 + lnlike_m_beta + lnlike_halflife

    return result


def lnprior(theta, inverted):
    """Log-prior on the parameters. Priors are taken to be scale-invariant, i.e. uniform on [0,2*pi]
    for angles and log-uniform for masses.
    """
    
    sigma,delta_m2_21,delta_m2_23,_,_,_,_ = theta

    for angle in theta[-4:]:
        if angle<0 or angle>2*np.pi:
            return -np.inf
        
    if sigma < 0 or delta_m2_21 < 0 or (delta_m2_23*(inverted - 0.5)) < 0:
        return -np.inf
    
    # make sure none of the masses are zero
    masses,success = basis_change(sigma,delta_m2_21,delta_m2_23)
    if success is False:
        return -np.inf
    if np.any(masses < 0):
        return -np.inf
    
    prior_sigma = -np.log(sigma)
    prior_m2_21 = -np.log(delta_m2_21)
    prior_m2_23 = -np.log(2.*(inverted - 0.5)*delta_m2_23)
    
    result =  prior_sigma + prior_m2_21 + prior_m2_23

    return result


def lnprob(theta, inverted, chi2_funcs, params):
    """Function to be passed to the MCMC sampler.
    """

    lp = lnprior(theta,inverted)
    ll = lnlike(theta,chi2_funcs,params,lp)

    return lp + ll


def mcmc(inverted=False, ncores=10, nwalkers=500, niter=1000, filename=None):
    """Run the MCMC sampler.

    :param inverted: use inverted ordering, defaults to False
    :type inverted: bool, optional
    :param ncores: number of CPU cores to use, defaults to 10
    :type ncores: int, optional
    :param nwalkers: number of walkers, defaults to 500
    :type nwalkers: int, optional
    :param niter: number of iterations, defaults to 1000
    :type niter: int, optional
    :param filename: path to the output file, defaults to None
    :type filename: str, optional
    :return: the array of samples
    :rtype: numpy.ndarray
    """

    np.seterr(invalid='ignore')

    if filename is None:
        filename = 'samples_{}_{}.npy'.format(['no', 'io'][inverted], nwalkers*niter)

    # load the chi-squared data from which the likelihood functions will be constructed
    data_path = '/'.join(__file__.split('/')[:-3]) + '/data/'
    chi2_m2_beta_func = load_endpoint_data(data_path)
    chi2_halflife_func = load_0vbb_data(data_path)
    chi2_osc_funcs = load_osc_data(inverted=inverted, data_dir=data_path)

    # construct arguments to be passed to the log-probability function
    chi2_funcs = [chi2_m2_beta_func] + [chi2_halflife_func] + list(chi2_osc_funcs)
    params = load_params(data_path)

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
    initial_mean = np.array([sigma_mean, delta_m2_21_mean, delta_m2_23_mean,\
                             theta_12_mean, theta_13_mean, alpha_21_mean, delta_minus_alpha31_mean])
    initial_err = np.array([sigma_err, delta_m2_21_err, delta_m2_23_err,\
                            theta_12_err, theta_13_err, alpha_21_err, delta_minus_alpha31_err])
    ndim = len(initial_mean)
    p0 = [np.random.normal(loc=initial_mean,scale=initial_err) for i in range(nwalkers)]

    # make sure we're not requesting more cores than there are available
    if ncores > cpu_count():
        print('Warning: {} cores requested but only {} available!'.format(ncores, cpu_count()))
        ncores = cpu_count()

    print('Running MCMC using {} cores'.format(ncores))
    with Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, \
                                        moves=emcee.moves.StretchMove(a=20), args=[inverted, chi2_funcs,params])
        
        print("Running burn-in...")
        p0,_,_ = sampler.run_mcmc(p0, 1000, progress=True)
        sampler.reset()

        print("Running production...")
        pos,prob,state = sampler.run_mcmc(p0, niter, progress=True)

    print('Saving the results...')
    samples = sampler.flatchain
    np.savetxt(filename, samples)

    print('Results saved to {}.'.format(filename))

    return samples