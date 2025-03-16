import numpy as np
from scipy.optimize import root


def basis_change(sigma,dm2_21,dm2_23):
    """Finds the 3 individual neutrino masses given the sum and mass squared differences.
    """
    
    # function whose roots give the individual neutrino masses
    func = lambda m: [m[0]+m[1]+m[2]-sigma,m[1]**2-m[0]**2-dm2_21,m[1]**2-m[2]**2-dm2_23]

    # try to find the roots with a naive guess
    sol = root(func,x0=[1e-1]*3,tol=1e-12)
    masses = sol.x
    success = sol.success

    # if that doesn't work, randomly adjust the guess until the roots are found or
    # the maximum number of iterations is exceeded (grid search or something more
    # sensible hasn't worked for me)
    iters = 0
    while success==False:
        sol = root(func,x0=np.random.normal(loc=1e-1,scale=5e-2,size=3),tol=1e-12)
        masses = sol.x
        success = sol.success
        iters += 1
        if iters>1e3:
            print('Failed to find roots of basis change function.')
            break

    return masses,success


def m_b(m1,m2,m3,s12,s13,c12,c13):
    """Electron antineutrino mass as a function of the masses and mixing angles.
    """
    return np.sqrt(m1**2*c12**2*c13**2 + m2**2*s12**2*c13**2 + m3**2*s13**2)


def m_bb(m1,m2,m3,s12,s13,c12,c13,alpha21,delta_minus_alpha31):
    """Effective Majorana mass as a function of the masses, mixing angles, and phases.
    """
    return np.abs(m1*c12**2*c13**2 + m2*s12**2*c13**2*np.exp(1j*alpha21) + m3*s13**2*np.exp(-1j*(delta_minus_alpha31)))


def T_half(m_bb,params):
    """Half life of the neutrinoless double beta decay given the effective Majorana mass.
    """
    m_e = 511e3 # electron mass in eV
    return 1./(m_bb**2*params['G_0v']*params['g_A']**4*params['M_0v']**2/m_e**2)


def model(theta):
    """Model that describes the posterior probability distribution to be sampled from.
    """
    sigma,delta_m2_21,delta_m2_23,theta12,theta13,alpha21,alpha31_minus_delta = theta
    masses,success = basis_change(sigma,delta_m2_21,delta_m2_23)
    if success is False:
        masses = np.zeros_like(masses)
    return m_bb(*masses,np.sin(theta12),np.sin(theta13),np.cos(theta12),np.cos(theta13),alpha21,alpha31_minus_delta)


def m2_no(m1,dm2_21):
    """Mass 2 as a function of mass 1 under normal ordering.
    """
    return np.sqrt(m1**2 + dm2_21)


def m3_no(m1,dm2_21,dm2_23):
    """Mass 3 as a function of mass 1 under normal ordering.
    """
    return np.sqrt(m1**2 + dm2_21 - dm2_23)


def m1_io(m3,dm2_23,dm2_21):
    """Mass 1 as a function of mass 3 under inverted ordering.
    """
    return np.sqrt(m3**2 + dm2_23 - dm2_21)


def m2_io(m3,dm2_23):
    """Mass 2 as a function of mass 3 under inverted ordering.
    """
    return np.sqrt(m3**2 + dm2_23)