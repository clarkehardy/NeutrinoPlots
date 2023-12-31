import numpy as np
from tqdm import tqdm
from util_funcs import *


def get_3sigma_range(m_lightest,params,inverted=False,nsamples=1e4):
    '''
    For a single value of the lightest mass, find the 3 sigma range of
    the effective Majorana mass.
    '''
    
    # get the allowed ranges from the params dictionary
    delta_m2_21_range = np.array(params['delta_m2_21'])[(2+2*inverted):(4+2*inverted)]
    delta_m2_23_range = np.array(params['delta_m2_23'])[(2+2*inverted):(4+2*inverted)]
    theta_12_range = np.array(params['theta_12'])[(2+2*inverted):(4+2*inverted)]*np.pi/180.
    theta_13_range = np.array(params['theta_13'])[(2+2*inverted):(4+2*inverted)]*np.pi/180.

    # sample values from within the 3 sigma allowed region
    nsamples = int(nsamples)
    delta_m2_21 = delta_m2_21_range[0] + np.random.beta(0.1,0.1,nsamples)*(delta_m2_21_range[1] - delta_m2_21_range[0])
    delta_m2_23 = delta_m2_23_range[0] + np.random.beta(0.1,0.1,nsamples)*(delta_m2_23_range[1] - delta_m2_23_range[0])
    theta_12 = theta_12_range[0] + np.random.beta(0.1,0.1,nsamples)*(theta_12_range[1] - theta_12_range[0])
    theta_13 = theta_13_range[0] + np.random.beta(0.1,0.1,nsamples)*(theta_13_range[1] - theta_13_range[0])
    alpha_21 = np.random.uniform(0,2*np.pi,nsamples)
    delta_minus_alpha31 = np.random.uniform(0,2*np.pi,nsamples)

    # initial limits to be replaced as more values are calculated
    m_bb_lower = 1e12
    m_bb_upper = 1e-12

    if inverted:
        for i in range(nsamples):
            # compute the effective Majorana mass for the inverted ordering
            m_1 = m1_io(m_lightest,delta_m2_23[i],delta_m2_21[i])
            m_2 = m2_io(m_lightest,delta_m2_23[i])
            m_betabeta = m_bb(m_1,m_2,m_lightest,np.sin(theta_12[i]),np.sin(theta_13[i]),\
                              np.cos(theta_12[i]),np.cos(theta_13[i]),alpha_21[i],delta_minus_alpha31[i])
            
            # replace the bounds if the new values fall outside them
            if m_betabeta > m_bb_upper:
                m_bb_upper = m_betabeta
            if m_betabeta < m_bb_lower:
                m_bb_lower = m_betabeta

    else:
        for i in range(nsamples):
            # compute the effective Majorana mass for the normal ordering
            m_2 = m2_no(m_lightest,delta_m2_21[i])
            m_3 = m3_no(m_lightest,delta_m2_21[i],delta_m2_23[i])
            m_betabeta = m_bb(m_lightest,m_2,m_3,np.sin(theta_12[i]),np.sin(theta_13[i]),\
                              np.cos(theta_12[i]),np.cos(theta_13[i]),alpha_21[i],delta_minus_alpha31[i])
            
            # replace the bounds if the new values fall outside them
            if m_betabeta > m_bb_upper:
                m_bb_upper = m_betabeta
            if m_betabeta < m_bb_lower:
                m_bb_lower = m_betabeta

    return m_bb_lower,m_bb_upper


def make_contours(params,inverted=False,npoints=100,nsamples=1e4):
    '''
    Get the upper and lower contours for the 3sigma allowed regions.
    '''

    m_lightest = np.logspace(-5,0,npoints)
    m_lower = np.zeros_like(m_lightest)
    m_upper = np.zeros_like(m_lightest)

    order = ['normal','inverted']
    print('Computing the 3 sigma allowed region for {} points for {} ordering...'.format(npoints,order[inverted]))
    for i,m_light in enumerate(tqdm(m_lightest)):
        m_l,m_u = get_3sigma_range(m_light,params,inverted,nsamples)
        m_lower[i] = m_l
        m_upper[i] = m_u

    return m_lightest,m_lower,m_upper