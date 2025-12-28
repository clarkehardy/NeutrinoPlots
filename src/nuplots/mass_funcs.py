import numpy as np
from scipy.optimize import root_scalar


def basis_change(sigma, dm2_21, dm2_23):
    """Finds the 3 individual neutrino masses given the sum of masses and two
    mass-squared differences. Determines hierarchy based on the sign of dm2_23.
    """
    if dm2_23 < 0:
        def residual(m1):
            try:
                m2 = np.sqrt(m1**2 + dm2_21)
                m3 = np.sqrt(m2**2 - dm2_23)
                return m1 + m2 + m3 - sigma
            except ValueError:
                return np.inf

        a, b = 1e-6, sigma
        fa, fb = residual(a), residual(b)
        if fa * fb > 0:
            return np.array([np.nan, np.nan, np.nan]), False

        result = root_scalar(residual, bracket=[a, b], method='brentq')
        if not result.converged:
            return np.array([np.nan, np.nan, np.nan]), False

        m1 = result.root
        m2 = np.sqrt(m1**2 + dm2_21)
        m3 = np.sqrt(m2**2 - dm2_23)
        return np.array([m1, m2, m3]), True

    elif dm2_23 > 0:
        def residual(m3):
            try:
                m2 = np.sqrt(m3**2 + dm2_23)
                m1 = np.sqrt(m2**2 - dm2_21)
                return m1 + m2 + m3 - sigma
            except ValueError:
                return np.inf

        a, b = 1e-6, sigma
        fa, fb = residual(a), residual(b)
        if fa * fb > 0:
            return np.array([np.nan, np.nan, np.nan]), False

        result = root_scalar(residual, bracket=[a, b], method='brentq')
        if not result.converged:
            return np.array([np.nan, np.nan, np.nan]), False

        m3 = result.root
        m2 = np.sqrt(m3**2 + dm2_23)
        m1 = np.sqrt(m2**2 - dm2_21)
        return np.array([m1, m2, m3]), True

    else:
        return np.array([np.nan, np.nan, np.nan]), False


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

def get_pmns_matrix(params):
    """Compute the PMNS matrix from mixing angles.
    
    :param params: Dictionary containing mixing angles in degrees
    :type params: dict
    :param theta_23: The mixing angle θ23 in degrees (default: 45.0)
    :type theta_23: float
    :return: the PMNS matrix
    :rtype: numpy.ndarray
    """
    theta_12 = np.deg2rad(params['theta_12'][0])
    theta_13 = np.deg2rad(params['theta_13'][0])
    theta_23 = np.deg2rad(params['theta_23'][0])
    
    R23 = np.array([[1, 0, 0],
                    [0, np.cos(theta_23), np.sin(theta_23)],
                    [0, -np.sin(theta_23), np.cos(theta_23)]])
    
    R13 = np.array([[np.cos(theta_13), 0, np.sin(theta_13)],
                    [0, 1, 0],
                    [-np.sin(theta_13), 0, np.cos(theta_13)]])
    
    r6 = np.array([[np.cos(theta_12), np.sin(theta_12), 0],
                    [-np.sin(theta_12), np.cos(theta_12), 0],
                    [0, 0, 1]])
    
    PMNS = R23 @ R13 @ r6
    
    return PMNS


def get_mass_ranges(params):
    """Get the minimum and maximum values of the three neutrino masses under
    either ordering scenario.

    :param params: neutrino oscillation parameters
    :type params: dict
    :return: array of minimum and array of maximum neutrino mass values
    :rtype: tuple
    """
    # Constants from experiment
    delta_m21_sq = params['delta_m2_21'][0]
    delta_m31_sq_NH = -params['delta_m2_23'][0]
    delta_m32_sq_IH = params['delta_m2_23'][0]
    sum_mass_limit = 0.12 # from Planck [eq. (1) in arXiv:1912.08208]

    m_lightest = np.concatenate(([0], np.linspace(1e-4, 0.1, 1000)))
    all_masses = []

    for m1 in m_lightest:
        # m1_, m2_, m3_ = get_masses_NH(m1)
        m2 = m2_no(m1, delta_m21_sq)
        m3 = m3_no(m1, delta_m21_sq, -delta_m31_sq_NH)
        if m1 + m2 + m3 <= sum_mass_limit:
            all_masses.append((m1, m2, m3))

    for m3 in m_lightest:
        # result = get_masses_IH(m3)
        m1 = m1_io(m3, -delta_m32_sq_IH, delta_m21_sq)
        m2 = m2_io(m3, -delta_m32_sq_IH)
        # if result:
        if not (np.isnan(m1) or np.isnan(m2)):
            # m1_, m2_, m3_ = result
            if m1 + m2 + m3 <= sum_mass_limit:
                all_masses.append((m1, m2, m3))

    arr = np.array(all_masses)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)

    return mins, maxs