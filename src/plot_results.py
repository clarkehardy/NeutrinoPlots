import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
from matplotlib import style
import numpy as np
from util_funcs import *
from prob_funcs import *
from contours import *
style.use('clarke-default')


def lobster_plot(samples_no=None,samples_io=None,npoints=100_000,nbins=200,params=None):

    # colorbar does not work well with tight_layout so
    # ensure that it is not enabled by default
    rcParams.update({'figure.autolayout': False})

    # sample from the posterior for normal ordering
    if samples_no is not None:
        ml_no = []
        mbb_no = []
        for theta in samples_no[np.random.randint(len(samples_no), size=npoints)]:
            masses,success = basis_change(*theta[:3])
            if success==False:
                continue
            ml_no.append(min(masses))
            mbb_no.append(model(theta))

    # sample from the posterior for inverted ordering
    if samples_io is not None:
        ml_io = []
        mbb_io = []
        for theta in samples_io[np.random.randint(len(samples_io), size=npoints)]:
            masses,success = basis_change(*theta[:3])
            if success==False:
                continue
            ml_io.append(min(masses))
            mbb_io.append(model(theta))

    if params is not None:
        m_lightest_no,m_lower_no,m_upper_no = make_contours(params,inverted=False,npoints=100,nsamples=1e4)
        m_lightest_io,m_lower_io,m_upper_io = make_contours(params,inverted=True,npoints=100,nsamples=1e4)

    # make the plot
    fig,axs = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    x_bins = np.logspace(-5,0,nbins)
    y_bins = np.logspace(-4,0,nbins)
    if samples_no is not None:
        h = axs[0].hist2d(ml_no,mbb_no,bins=(x_bins,y_bins),cmin=1,cmap='inferno',norm=LogNorm())
    if samples_io is not None:
        h = axs[1].hist2d(ml_io,mbb_io,bins=(x_bins,y_bins),cmin=1,cmap='inferno',norm=LogNorm())
    labels = ['Normal ordering','Inverted ordering']
    if params is not None:
        axs[0].plot(m_lightest_no,m_lower_no,'k')
        axs[0].plot(m_lightest_no,m_upper_no,'k')
        axs[1].plot(m_lightest_io,m_lower_io,'k')
        axs[1].plot(m_lightest_io,m_upper_io,'k')
    for i,ax in enumerate(axs):
        ax.text(2e-5,4e-1,labels[i])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r'$m_{{{}}}$ [eV]'.format(2*i+1))
        ax.set_ylim([1e-4,1])
        ax.set_xlim([1e-5,1])
        ax.grid()
    axs[0].set_ylabel(r'$m_{\beta\beta}$ [eV]')
    axs[1].set_yticklabels([])
    fig.suptitle('Marginalized posterior distributions')
    fig.colorbar(h[3], ax=axs.ravel().tolist(),label='Probability density')

    return fig,axs