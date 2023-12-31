import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
from matplotlib import style
import numpy as np
from utils.util_funcs import *
from lobster.prob_funcs import *
from lobster.contours import *
style.use('clarke-default')


def density(samples_no=None,samples_io=None,npoints=100_000,nbins=200,params=None,cmap='inferno'):
    '''
    Make the lobster plot showing the probability density in the allowed regions.
    '''
    # colorbar does not work well with tight_layout so
    # ensure that it is not enabled by default
    rcParams.update({'figure.autolayout': False})
    x_bins = np.logspace(-5,0,nbins)
    y_bins = np.logspace(-4,0,nbins)
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
        # histogram the results and normalize by the log of the bin edges
        h_no,x_edges,y_edges = np.histogram2d(ml_no,mbb_no,bins=(x_bins,y_bins))
        x_diff = np.log(x_edges[1]/x_edges[0])
        y_diff = np.log(y_edges[1]/y_edges[0])
        h_no = h_no/(x_diff*y_diff*np.sum(h_no))
        if np.amax(h_no)>vmax:
            vmax = np.amax(h_no)
        X,Y = np.meshgrid(x_edges,y_edges)

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
        # histogram the results and normalize by the log of the bin edges
        h_io,x_edges,y_edges = np.histogram2d(ml_io,mbb_io,bins=(x_bins,y_bins))
        x_diff = np.log(x_edges[1]/x_edges[0])
        y_diff = np.log(y_edges[1]/y_edges[0])
        h_io = h_io/(x_diff*y_diff*np.sum(h_io))
        if np.amax(h_io)>vmax:
            vmax = np.amax(h_io)
        X,Y = np.meshgrid(x_edges,y_edges)

    if params is not None:
        m_lightest_no,m_lower_no,m_upper_no = get_contours(params,inverted=False,npoints=100,nsamples=1e5)
        m_lightest_io,m_lower_io,m_upper_io = get_contours(params,inverted=True,npoints=100,nsamples=1e5)

    # make the plot
    fig,axs = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    if params is not None:
        colors = plt.get_cmap(cmap)
        axs[0].fill_between(m_lightest_no,m_lower_no,m_upper_no,color=colors(0),alpha=0.8)
        axs[1].fill_between(m_lightest_io,m_lower_io,m_upper_io,color=colors(0),alpha=0.8)
    if samples_no is not None:
        im = axs[0].pcolormesh(X,Y,h_no.T,cmap=cmap,norm=LogNorm(vmin=10**(np.log10(vmax)-3),vmax=vmax))
    if samples_io is not None:
        im = axs[1].pcolormesh(X,Y,h_io.T,cmap=cmap,norm=LogNorm(vmin=10**(np.log10(vmax)-3),vmax=vmax))
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
    fig.colorbar(im, ax=axs.ravel().tolist(),label='Probability density')

    return fig,axs


def vanilla(params,npoints=1e4,nsamples=200):
    '''
    Make the vanilla lobster plot showing the allowed regions.
    '''

    m_lightest_no,m_lower_no,m_upper_no = get_contours(params,inverted=False,npoints=npoints,nsamples=nsamples)
    m_lightest_io,m_lower_io,m_upper_io = get_contours(params,inverted=True,npoints=npoints,nsamples=nsamples)

    fig,ax = plt.subplots()
    ax.fill_between(m_lightest_no,m_lower_no,m_upper_no,color='red',alpha=0.5,label='Normal ordering')
    ax.fill_between(m_lightest_io,m_lower_io,m_upper_io,color='blue',alpha=0.5,label='Inverted ordering')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-4,1e0])
    ax.set_xlim([1e-5,1e0])
    ax.set_ylabel(r'$m_{\beta\beta}$ [eV]')
    ax.set_xlabel(r'$m_\mathrm{lightest}$ [eV]')
    ax.set_title(r'3$\sigma$ allowed regions for the effective Majorana mass')
    ax.legend(loc='upper left')

    return fig,ax