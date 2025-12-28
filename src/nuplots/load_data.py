import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv
import yaml


def load_osc_data(inverted=False, data_dir='../data', plot=False):
    """Load the one-dimensional projections of the three-neutrino oscillation parameter fit from a global analysis of
    solar, atmospheric, accelerator, and reactor data provided by nu-fit.org.

    :param inverted: whether to use inverted neutrino mass ordering
    :type inverted: bool
    :param data_dir: directory in which the oscillation parameters are stored
    :type data_dir: str
    :param plot: whether to plot the oscillation parameter fits
    :type plot: bool
    :return: interpolating functions for chi^2 values. If plot is True, also returns figure and axes objects.
    :rtype: tuple
    """

    # get the correct file for normal or inverted ordering
    if inverted:
        path = data_dir + '/v60.release-TByes-IO.txt'
    else:
        path = data_dir + '/v60.release-TByes-NO.txt'

    # prefixes to identify the correct sections of the file
    prefixes = ['T13','T12','T23','DCP','DMS','DMA']

    # load the raw data
    sin2_theta12_vals = []
    sin2_theta13_vals = []
    sin2_theta23_vals = []
    delta_cp_vals = []
    delta_m2_21_vals = []
    delta_m2_32_vals = []
    sin2_theta12_chi2 = []
    sin2_theta13_chi2 = []
    sin2_theta23_chi2 = []
    delta_cp_chi2 = []
    delta_m2_21_chi2 = []
    delta_m2_32_chi2 = []

    lists = [sin2_theta13_vals,sin2_theta12_vals,sin2_theta23_vals,delta_cp_vals,delta_m2_21_vals,delta_m2_32_vals]
    chi2s = [sin2_theta13_chi2,sin2_theta12_chi2,sin2_theta23_chi2,delta_cp_chi2,delta_m2_21_chi2,delta_m2_32_chi2]

    curr_ind = -1

    with open(path,'r') as infile:
        reader = csv.reader(infile,delimiter=' ')
        for i,line in enumerate(reader):
            if curr_ind != -1 and len(line)>1 and line[-1] != 'Delta_chi^2':
                lists[curr_ind].append(float(line[-2]))
                chi2s[curr_ind].append(float(line[-1]))
            if len(line)>0:
                if line[1] in prefixes:
                    curr_ind += 1

    # save as numpy arrays and modify to the format used in this analysis
    sin2_theta12_vals = np.array(sin2_theta12_vals)
    sin2_theta13_vals = np.array(sin2_theta13_vals)
    # data provided in the tables is for log10(delta_m2_21)
    delta_m2_21_vals = 10**np.array(delta_m2_21_vals)
    # data provided in the tables is for delta_m2_32/1e-3
    delta_m2_23_vals = -1e-3*np.array(delta_m2_32_vals)
    sin2_theta12_chi2 = np.array(sin2_theta12_chi2)
    sin2_theta13_chi2 = np.array(sin2_theta13_chi2)
    delta_m2_21_chi2 = np.array(delta_m2_21_chi2)
    delta_m2_23_chi2 = np.array(delta_m2_32_chi2)

    # interpolating functions for the chi^2 tables, with data outside the range set to infinity
    chi2_sin2_theta12_func = interp1d(sin2_theta12_vals,sin2_theta12_chi2,bounds_error=False,fill_value=np.inf)
    chi2_sin2_theta13_func = interp1d(sin2_theta13_vals,sin2_theta13_chi2,bounds_error=False,fill_value=np.inf)
    chi2_delta_m2_21_func = interp1d(delta_m2_21_vals,delta_m2_21_chi2,bounds_error=False,fill_value=np.inf)
    chi2_delta_m2_23_func = interp1d(delta_m2_23_vals,delta_m2_23_chi2,bounds_error=False,fill_value=np.inf)

    if plot:
        # update the lists to the new arrays for plotting
        lists = [sin2_theta12_vals,sin2_theta13_vals,delta_m2_21_vals,delta_m2_23_vals]
        chi2s = [sin2_theta12_chi2,sin2_theta13_chi2,delta_m2_21_chi2,delta_m2_23_chi2]

        labels = [r'$\sin^2{\theta_{12}}$',r'$\sin^2{\theta_{13}}$',r'$\Delta m_{21}^2$ [eV$^2$]',r'$\Delta m_{23}^2$ [eV$^2$]']
        ordering = ['normal ordering','inverted ordering']

        # make fig,axs to be returned in addition to the interpolating functions
        fig,axs = plt.subplots(2,2,figsize=(8,7))
        for i,ax in enumerate(axs.flatten()):
            ax.plot(lists[i],chi2s[i])
            ax.set_ylabel(r'$\Delta\chi^2$')
            ax.set_xlabel(labels[i])
            ax.grid()
        fig.suptitle('Fit results from oscillation data for '+ordering[inverted])

        return chi2_sin2_theta12_func,chi2_sin2_theta13_func,chi2_delta_m2_21_func,chi2_delta_m2_23_func,fig,axs

    # return the interpolating functions
    return chi2_sin2_theta12_func,chi2_sin2_theta13_func,chi2_delta_m2_21_func,chi2_delta_m2_23_func


def load_endpoint_data(data_dir='../data', plot=False):
    """Load data from beta spectrum endpoint measurements for the effective electron neutrino mass.

    :param data_dir: directory in which the oscillation parameters are stored
    :type data_dir: str
    :param plot: whether to plot the endpoint measurement fit
    :type plot: bool
    :return: interpolating function for chi^2 values. If plot is True, also returns figure and axes objects.
    :rtype: scipy.interpolate.interp1d or tuple
    """

    # path to the supplementary data provided by KATRIN
    path = data_dir + '/41567_2021_1463_MOESM5_ESM.txt'

    # load the data
    m2_beta_chi2 = []
    m2_beta_vals = []

    with open(path,'r') as infile:
        reader = csv.reader(infile,delimiter='\t')
        for i in range(6):
            next(reader,None)
        for line in reader:
            m2_beta_vals.append(float(line[0]))
            m2_beta_chi2.append(float(line[2]))

    m2_beta_vals = np.array(m2_beta_vals)
    m2_beta_chi2 = np.array(m2_beta_chi2)

    # make the interpolating function
    chi2_m2_beta_func = interp1d(m2_beta_vals,m2_beta_chi2,bounds_error=False,fill_value=np.inf)

    if plot:
        fig,ax = plt.subplots()
        ax.plot(m2_beta_vals,m2_beta_chi2)
        ax.set_xlabel(r'$m_\beta^2$ [eV$^2$]')
        ax.set_ylabel(r'$\Delta\chi^2$')
        ax.set_title('Fit results from beta spectrum endpoint measurement')
        ax.grid()

        return chi2_m2_beta_func, fig, ax
    
    # return the interpolating function
    return chi2_m2_beta_func


def load_0vbb_data(data_dir='../data', plot=False):
    """Load data from neutrinoless double beta decay searches which place limits on the effective Majorana mass.

    :param data_dir: directory in which the oscillation parameters are stored
    :type data_dir: str
    :param plot: whether to plot the 0vbb limit fit
    :type plot: bool
    :return: interpolating function for chi^2 values. If plot is True, also returns figure and axes objects.
    :rtype: scipy.interpolate.interp1d or tuple
    """

    # path to the supplementary data provided by KamLAND-Zen
    path = data_dir + '/SupplementaryData_KamLANDZen_20220304.txt'

    halflife_vals = []
    halflife_chi2 = []

    with open(path,'r') as infile:
        reader = csv.reader(infile,delimiter=' ')
        for i in range(7):
            next(reader,None)
        for line in reader:
            halflife_vals.append(float(line[0]))
            halflife_chi2.append(float(line[1]))

    halflife_vals = np.array(halflife_vals)*1e26
    halflife_chi2 = np.array(halflife_chi2)

    # make the interpolating function
    chi2_halflife_func = interp1d(halflife_vals,halflife_chi2,bounds_error=False,fill_value=(np.inf,0))

    if plot:
        fig,ax = plt.subplots()
        ax.plot(halflife_vals/1e26,halflife_chi2)
        ax.set_xlabel(r'T$_{1/2}$ [10$^{26}$ years]')
        ax.set_ylabel(r'$\Delta\chi^2$')
        ax.set_title(r'Half-life limit from $0\nu\beta\beta$ search')
        ax.grid()

        return chi2_halflife_func,fig,ax
    
    # return the interpolating function
    return chi2_halflife_func


def load_params(data_dir='../data'):
    """Load parameters used to calculate the effective Majorana mass from the 0vbb
    half life, including the phase space factor, matrix element, and axial-vector
    coupling constant.

    :param data_dir: directory in which the oscillation parameters are stored
    :type data_dir: str
    :return: dictionary containing the parameters
    :rtype: dict
    """

    path = data_dir + '/params.yaml'

    with open(path,'r') as infile:
        params = yaml.safe_load(infile)

    return params


def load_fermion_masses(data_dir='../data'):
    """Load the fermion masses.

    :param data_dir: directory in which the fermion masses are stored
    :type data_dir: str
    :return: dictionary containing the masses
    :rtype: dict
    """

    path = data_dir + '/fermions.yaml'

    with open(path,'r') as infile:
        fermion_masses = yaml.safe_load(infile)

    for key in fermion_masses.keys():
        fermion_masses[key] = float(fermion_masses[key])

    return fermion_masses


def load_decay_chain(which='U-238', data_dir='../data'):
    """Load decay chain data.

    :param which: which decay chain to load
    :type which: str
    :param data_dir: directory in which the decay chain is stored
    :type data_dir: str, optional
    :return: dictionary containing the decay chain
    :rtype: dict
    """

    with open(data_dir + '/' + which.replace('-', '').lower() + '_chain.yaml', 'r') as f:
        chain = yaml.safe_load(f)

    for isotope in chain:
        if isotope['br'] is None:
            isotope['br'] = np.nan

    return chain
