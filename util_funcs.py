import numpy as np
from scipy.optimize import fsolve

def basis_change(sigma,dm2_21,dm2_23):
    '''
    Returns a function that can be minimized numerically to find the 3 masses given the
    mass squared differences and sum of masses.
    '''
    def func(m):
        '''
        Function to minimize to invert the basis change.
        '''
        return [m[0]+m[1]+m[2]-sigma,m[1]**2-m[0]**2-dm2_21,m[1]**2-m[2]**2-dm2_23]
    return func

def m_b(m1,m2,m3,s12,s13,c12,c13):
    '''
    Electron antineutrino mass as a function of the masses and mixing angles.
    '''
    return np.sqrt(m1**2*c12**2*c13**2 + m2**2*s12**2*c13**2 + m3**2*s13**2)


def m_bb(m1,m2,m3,s12,s13,c12,c13,alpha21,delta_minus_alpha31):
    '''
    Effective Majorana mass as a function of the masses, mixing angles, and phases.
    '''
    return np.abs(m1*c12**2*c13**2 + m2*s12**2*c13**2*np.exp(1j*alpha21) + m3*s13**2*np.exp(-1j*(delta_minus_alpha31)))


def T_half(m_bb,params):
    '''
    Half life of the neutrinoless double beta decay given the effective Majorana mass.
    '''
    m_e = 511e3 # electron mass in eV
    return 1./(m_bb**2*params['G_0v']*params['g_A']**4*params['M_0v']**2/m_e**2)


def model(theta):
    '''
    Model that describes the posterior probability distribution to be sampled from.
    '''
    sigma,delta_m2_21,delta_m2_23,theta12,theta13,alpha21,alpha31_minus_delta = theta
    m1,m2,m3 = fsolve(basis_change(sigma,delta_m2_21,delta_m2_23),x0=[1e-1]*3,xtol=1e-12)
    return m_bb(m1,m2,m3,np.sin(theta12),np.sin(theta13),np.cos(theta12),np.cos(theta13),alpha21,alpha31_minus_delta)