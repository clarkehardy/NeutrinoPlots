from nuplots.load_data import *
from nuplots.utils import *


def lnlike(theta,chi2_funcs,params,ln_prior):
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


def lnprior(theta,inverted):
    """Log-prior on the parameters. Priors are taken to be scale-invariant, i.e. uniform on [0,2*pi]
    for angles and log-uniform for masses.
    """
    
    sigma,delta_m2_21,delta_m2_23,_,_,_,_ = theta

    for angle in theta[-4:]:
        if angle<0 or angle>2*np.pi:
            return -np.inf
        
    if sigma<0 or delta_m2_21<0 or (delta_m2_23*(inverted-0.5))<0:
        return -np.inf
    
    # make sure none of the masses are zero
    masses,success = basis_change(sigma,delta_m2_21,delta_m2_23)
    if success is False:
        return -np.inf
    if np.any(masses < 0):
        return -np.inf
    
    prior_sigma = -np.log(sigma)
    prior_m2_21 = -np.log(delta_m2_21)
    prior_m2_23 = -np.log(2.*(inverted-0.5)*delta_m2_23)
    
    result =  prior_sigma + prior_m2_21 + prior_m2_23

    return result


def lnprob(theta,inverted,chi2_funcs,params):
    """Function to be passed to the MCMC sampler.
    """

    lp = lnprior(theta,inverted)
    ll = lnlike(theta,chi2_funcs,params,lp)

    return lp + ll