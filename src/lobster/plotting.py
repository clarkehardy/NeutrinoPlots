import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,LinearSegmentedColormap
from matplotlib import rcParams
import numpy as np
from scipy.stats import gaussian_kde
from utils.util_funcs import *
from lobster.prob_funcs import *
from lobster.contours import *


def density(samples_no=None,samples_io=None,npoints=100_000,nbins=200,params=None,\
            style='hist',cmap='magma_r',data_save_path=None):
    """Make the lobster plot showing the probability density in the allowed regions.
    """
    if style not in ['hist', 'kde']:
        print('Error: plotting style not recognized!')
        return

    # colorbar does not work well with tight_layout so
    # ensure that it is not enabled by default
    rcParams.update({'figure.autolayout': False})
    x_bins = np.logspace(-5,0,nbins + 1)
    y_bins = np.logspace(-4,0,nbins + 1)
    X,Y = np.meshgrid(x_bins[1:], y_bins[1:])
    vmax = 1e-12

    # sample from the posterior for normal ordering
    if samples_no is not None:
        ml_no = []
        mbb_no = []
        print('Sampling the posterior distribution for normal ordering...')
        for theta in tqdm(samples_no[np.random.randint(len(samples_no),size=npoints)]):
            masses,success = basis_change(*theta[:3])
            if success==False:
                continue
            ml_no.append(min(masses))
            mbb_no.append(model(theta))
        ml_no = np.array(ml_no)
        if np.any(ml_no < 0):
            print('Warning: {} samples had negative masses!'.format(np.sum(ml_no < 0)))
            ml_no[ml_no < 0] = 1e-100
        if style == 'hist':
            # histogram the results and normalize by the log of the bin edges
            h_no,x_edges,y_edges = np.histogram2d(ml_no,mbb_no,bins=(x_bins,y_bins))
            x_diff = np.log(x_edges[1]/x_edges[0])
            y_diff = np.log(y_edges[1]/y_edges[0])
            h_no = h_no/(x_diff*y_diff*np.sum(h_no))
            h_no = h_no.T
        else:
            values = np.vstack([np.log10(ml_no), np.log10(mbb_no)])
            values[np.isnan(values)] = 0
            kde = gaussian_kde(values)
            h_no = kde(np.vstack([np.log10(X.ravel()), np.log10(Y.ravel())])).reshape(X.shape)
        if np.amax(h_no)>vmax:
            vmax = np.amax(h_no)

    # sample from the posterior for inverted ordering
    if samples_io is not None:
        ml_io = []
        mbb_io = []
        print('Sampling the posterior distribution for inverted ordering...')
        for theta in tqdm(samples_io[np.random.randint(len(samples_io), size=npoints)]):
            masses,success = basis_change(*theta[:3])
            if success==False:
                continue
            ml_io.append(min(masses))
            mbb_io.append(model(theta))
        ml_io = np.array(ml_io)
        if np.any(ml_io < 0):
            print('Warning: {} samples had negative masses!'.format(np.sum(ml_io < 0)))
            ml_io[ml_io < 0] = 1e-100
        if style == 'hist':
            # histogram the results and normalize by the log of the bin edges
            h_io,x_edges,y_edges = np.histogram2d(ml_io,mbb_io,bins=(x_bins,y_bins))
            x_diff = np.log(x_edges[1]/x_edges[0])
            y_diff = np.log(y_edges[1]/y_edges[0])
            h_io = h_io/(x_diff*y_diff*np.sum(h_io))
            h_io = h_io.T
        else:
            values = np.vstack([np.log10(ml_io), np.log10(mbb_io)])
            values[np.isnan(values)] = 0
            kde = gaussian_kde(values)
            h_io = kde(np.vstack([np.log10(X.ravel()), np.log10(Y.ravel())])).reshape(X.shape)
        if np.amax(h_io)>vmax:
            vmax = np.amax(h_io)

    if params is not None:
        m_lightest_no,m_lower_no,m_upper_no = get_contours(params,inverted=False,npoints=nbins,nsamples=1e5)
        m_lightest_io,m_lower_io,m_upper_io = get_contours(params,inverted=True,npoints=nbins,nsamples=1e5)
        mask_no = (Y <= np.interp(X,m_lightest_no,m_upper_no)) & (Y >= np.interp(X,m_lightest_no,m_lower_no))
        mask_io = (Y <= np.interp(X,m_lightest_io,m_upper_io)) & (Y >= np.interp(X,m_lightest_io,m_lower_io))
    else:
        mask_no = np.ones(h_no.shape, dtype=bool)
        mask_io = np.ones(h_no.shape, dtype=bool)

    if data_save_path:
        if not data_save_path.endswith('.npz'):
            data_save_path = data_save_path + '.npz'
        save_dict = {}
        if samples_no is not None or samples_io is not None:
            save_dict['m_lightest'] = X
            save_dict['m_betabeta'] = Y
        if samples_no is not None:
            save_dict['prob_no'] = h_no
        if samples_io is not None:
            save_dict['prob_io'] = h_io
        if params is not None:
            save_dict.update({'m_lightest_no': m_lightest_no,
                              'm_lower_no': m_lower_no,
                              'm_upper_no': m_upper_no,
                              'm_lightest_io': m_lightest_io,
                              'm_lower_io': m_lower_io,
                              'm_upper_io': m_upper_io})
        np.savez(data_save_path, **save_dict)
        print('Plot data saved to ' + data_save_path)

    # make the plot
    fig,axs = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    colors = plt.get_cmap(cmap)
    cmap_reduced = LinearSegmentedColormap.from_list('',colors(np.linspace(0.05,1,1000)))
    if params is not None:
        axs[0].fill_between(m_lightest_no,m_lower_no,m_upper_no,color=colors(0),alpha=0.8)
        axs[1].fill_between(m_lightest_io,m_lower_io,m_upper_io,color=colors(0),alpha=0.8)
        axs[0].plot(m_lightest_no, m_lower_no, color='k', lw=1.5)
        axs[0].plot(m_lightest_no, m_upper_no, color='k', lw=1.5)
        axs[1].plot(m_lightest_io, m_lower_io, color='k', lw=1.5)
        axs[1].plot(m_lightest_io, m_upper_io, color='k', lw=1.5)
    if samples_no is not None:
        h_no[~mask_no] = 0
        im = axs[0].pcolormesh(X,Y,h_no,cmap=cmap_reduced,norm=LogNorm(vmin=10**(np.log10(vmax)-3),vmax=vmax))
    if samples_io is not None:
        h_io[~mask_io] = 0
        im = axs[1].pcolormesh(X,Y,h_io,cmap=cmap_reduced,norm=LogNorm(vmin=10**(np.log10(vmax)-3),vmax=vmax))
    labels = ['Normal ordering','Inverted ordering']
    for i,ax in enumerate(axs):
        ax.text(2e-5,4e-1,labels[i])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r'$m_{{{}}}$ [eV]'.format(2*i+1))
        ax.set_ylim([1e-4,1])
        ax.set_xlim([1e-5,1])
    axs[0].set_ylabel(r'$m_{\beta\beta}$ [eV]')
    axs[1].set_yticklabels([])
    fig.suptitle('Marginalized posterior probability for the effective Majorana mass')
    if samples_no is not None or samples_io is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(),label='Probability density')

    return fig,axs


def vanilla(params, npoints=200, nsamples=1e4, sum=False):
    """Make the vanilla lobster plot showing the allowed regions, with the
    x-axis showing either the lightest mass or the sum of masses.
    """

    m_lightest_no,m_lower_no,m_upper_no = get_contours(params, inverted=False, npoints=npoints, \
                                                       nsamples=nsamples, sum=sum)
    m_lightest_io,m_lower_io,m_upper_io = get_contours(params, inverted=True, npoints=npoints, \
                                                       nsamples=nsamples, sum=sum)

    fig,ax = plt.subplots()
    ax.fill_between(m_lightest_no,m_lower_no,m_upper_no,color='red',alpha=0.5,lw=0,\
                    label='Normal ordering')
    ax.fill_between(m_lightest_io,m_lower_io,m_upper_io,color='blue',alpha=0.5,lw=0,\
                    label='Inverted ordering')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-4,1e0])
    ax.set_xlim([1e-5,1e0])
    ax.set_ylabel(r'$m_{\beta\beta}$ [eV]')
    ax.set_xlabel(r'$m_\mathrm{lightest}$ [eV]')
    if sum:
        ax.set_ylim([1e-4,1e0])
        ax.set_xlim([5e-2,1e0])
        ax.set_xlabel(r'$\Sigma$ [eV]')
    ax.set_title(r'3$\sigma$ allowed regions for the effective Majorana mass')
    ax.legend(loc='upper left')

    return fig,ax