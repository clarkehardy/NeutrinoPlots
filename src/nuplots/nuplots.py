import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgba, to_hex
from matplotlib.lines import Line2D
from matplotlib import color_sequences
from scipy.stats import gaussian_kde
import feynman
from nuplots.plot_utils import CompositePatch, HandlerCompositePatch, set_fonts
from nuplots.mass_funcs import get_mass_ranges, get_pmns_matrix
from nuplots.lobster import get_contours, model, basis_change


def plot_in_font(func):
    """Decorator function to allow other plotting functions to be called in an
    rcParams context window.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        font = kwargs.get('font', None)
        if font is None:
            return func(*args, **kwargs)
        serif = kwargs.get('serif', False)
        font_path = '/'.join(__file__.split('/')[:-3]) + '/fonts/' + font.replace(' ', '') + '/'
        rcparams = set_fonts(font, font_path, serif=serif)
        with plt.rc_context(rcparams):
            return func(*args, **kwargs)
    return wrapper


@plot_in_font
def lobster_density(samples_no=None, samples_io=None, npoints=100_000, nbins=200, params=None,\
                    sum=False, style='hist', cmap='magma_r', data_save_path=None, save_path=None, \
                    font=None, serif=False):
    """Make the lobster plot showing the probability density in the allowed regions.
    """
    if style not in ['hist', 'kde']:
        print('Error: plotting style not recognized!')
        return
    
    if sum:
        func = np.sum
    else:
        func = np.amin

    # colorbar does not work well with tight_layout so
    # ensure that it is not enabled by default
    x_bins = np.logspace([-5, np.log10(5e-2)][int(sum)], 0, nbins + 1)
    y_bins = np.logspace(-4, 0, nbins + 1)
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
            ml_no.append(func(masses))
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
            ml_io.append(func(masses))
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
        m_lightest_no,m_lower_no,m_upper_no = get_contours(params,inverted=False,npoints=nbins,nsamples=1e5,sum=sum)
        m_lightest_io,m_lower_io,m_upper_io = get_contours(params,inverted=True,npoints=nbins,nsamples=1e5,sum=sum)
        mask_no = (Y <= np.interp(X,m_lightest_no,m_upper_no)) & (Y >= np.interp(X,m_lightest_no,m_lower_no))
        mask_io = (Y <= np.interp(X,m_lightest_io,m_upper_io)) & (Y >= np.interp(X,m_lightest_io,m_lower_io))
    else:
        mask_no = np.ones(h_no.shape, dtype=bool)
        mask_io = np.ones(h_no.shape, dtype=bool)

    if data_save_path:
        if not data_save_path.endswith('.npz'):
            data_save_path = data_save_path + '.npz'
        save_dict = {}
        if sum:
            m_label = 'sigma'
        else:
            m_label = 'm_lightest'
        if samples_no is not None or samples_io is not None:
            save_dict[m_label] = X
            save_dict['m_betabeta'] = Y
        if samples_no is not None:
            save_dict['prob_no'] = h_no
        if samples_io is not None:
            save_dict['prob_io'] = h_io
        if params is not None:
            save_dict.update({m_label + '_no': m_lightest_no,
                              'm_lower_no': m_lower_no,
                              'm_upper_no': m_upper_no,
                              m_label + '_io': m_lightest_io,
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
        ax.set_yscale('log')
        ax.set_xscale('log')
        if sum:
            ax.text(6e-2,4e-1,labels[i])
            ax.set_xlabel(r'$\Sigma$ [eV]')
            ax.set_xlim([5e-2,1])
        else:
            ax.text(2e-5,4e-1,labels[i])
            ax.set_xlabel(r'$m_{{{}}}$ [eV]'.format(2*i+1))
            ax.set_xlim([1e-5,1])
        ax.set_ylim([1e-4,1])
    axs[0].set_ylabel(r'$m_{\beta\beta}$ [eV]')
    axs[1].set_yticklabels([])
    fig.suptitle('Marginalized posterior probability for the effective Majorana mass')
    if samples_no is not None or samples_io is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(),label='Probability density')

    if save_path:
        fig.savefig(save_path)

    return fig, axs


@plot_in_font
def lobster_vanilla(params, npoints=200, nsamples=1e4, sum=False, save_path=None, \
                    colors=None, font=None, serif=False):
    """Make the vanilla lobster plot showing the allowed regions, with the
    x-axis showing either the lightest mass or the sum of masses.
    """

    if colors is None:
        colors = [color_sequences['tab10'][3], color_sequences['tab10'][0]]

    m_lightest_no,m_lower_no,m_upper_no = get_contours(params, inverted=False, npoints=npoints, \
                                                       nsamples=nsamples, sum=sum)
    m_lightest_io,m_lower_io,m_upper_io = get_contours(params, inverted=True, npoints=npoints, \
                                                       nsamples=nsamples, sum=sum)

    fig,ax = plt.subplots(figsize=(4, 3), layout='constrained')
    ax.fill_between(m_lightest_no, m_lower_no, m_upper_no, color=colors[0], alpha=0.5, lw=0, \
                    label='Normal ordering')
    ax.fill_between(m_lightest_io, m_lower_io, m_upper_io, color=colors[1], alpha=0.5, lw=0, \
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
    ax.set_title(r'3$\sigma$ allowed regions for $m_{\beta\beta}$')
    ax.legend(loc='upper left')

    if save_path:
        fig.savefig(save_path)

    return fig, ax


@plot_in_font
def oscillation(params, save_path=None, colors=None, font=None):
    """Make a plot showing the oscillation probability as a function of L/E.

    :param params: dictionary of neutrino oscillation parameters
    :type params: dict
    :param save_path: path where the figure should be saved, defaults to None
    :type save_path: str, optional
    """
    if colors is None:
        colors = color_sequences['tab10']
    nu_colors = [colors[3], colors[8], colors[0]]

    delta_m21_sq = params['delta_m2_21'][0]
    delta_m31_sq_NH = -params['delta_m2_23'][0]

    U = get_pmns_matrix(params)
    L_by_E = np.logspace(2, np.log10(6e4), 1000)

    def flavor_fractions(L_by_E):
        dm = np.array([0, delta_m21_sq, delta_m31_sq_NH])
        psi_mass = np.zeros((3, len(L_by_E)), dtype=np.complex128)
        for i in range(3):
            phase = np.exp(-1j * dm[i] * 1.267 * L_by_E)
            psi_mass[i] = U[0, i] * phase

        psi_flavor = U @ psi_mass
        probs = np.abs(psi_flavor)**2
        return probs

    fractions = flavor_fractions(L_by_E)
    labels = [r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$']

    fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
    ax.stackplot(L_by_E, fractions, labels=labels, colors=nu_colors, alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlim(L_by_E[0], L_by_E[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('$L/E$ [km/GeV]')
    ax.set_ylabel('Flavor fraction')
    ax.legend(loc='lower left')

    if save_path:
        fig.savefig(save_path)

    return fig, ax

@plot_in_font
def mass_ordering(params, save_path=None, colors=None, font=None, serif=False):
    """Make a plot showing the ordering of the neutrino masses under both
    ordering scenarios.

    :param params: dictionary of neutrino oscillation parameters
    :type params: dict
    :param save_path: path where the figure should be saved, defaults to None
    :type save_path: str, optional
    """
    if colors is None:
        colors = color_sequences['tab10']
    nu_colors = [colors[3], colors[8], colors[0]]

    # mass-squared differences are not plotted to scale
    dm2_sol = 1
    dm2_atm = 6
    U_PMNS = get_pmns_matrix(params)
    U_inv = np.conj(U_PMNS).T
    m2_1 = np.abs(U_inv[:,0])**2
    m2_2 = np.abs(U_inv[:,1])**2
    m2_3 = np.abs(U_inv[:,2])**2
    no_states = np.array((0, dm2_sol, dm2_sol + dm2_atm))
    io_states = np.array((0, dm2_atm, dm2_atm + dm2_sol))
    mass_labels = np.array(['$m_1^2$', '$m_2^2$', '$m_3^2$'])
    ordering_labels = ['Normal ordering', 'Inverted ordering']
    mass_states = [m2_1, m2_2, m2_3]
    states = [no_states, io_states]
    indices = [np.arange(3), np.array((2, 0, 1))]
    arrow_pos = 0.5
    shrink = 5
    height = 0.2
    fontsize = plt.rcParams['font.size']

    fig, ax = plt.subplots(1, 2, figsize=(4, 4), layout='constrained')
    for i in range(2):
        ax[i].barh(states[i], m2_1[indices[i]], height=height, color=nu_colors[0], label=r'$\nu_e$')
        ax[i].barh(states[i], m2_2[indices[i]], left=m2_1[indices[i]], height=height, color=nu_colors[1], label=r'$\nu_\mu$')
        ax[i].barh(states[i], m2_3[indices[i]], left=m2_1[indices[i]] + m2_2[indices[i]], height=height, color=nu_colors[2], label=r'$\nu_\tau$')
        ax[i].set_yticks(states[i])
        ax[i].set_yticklabels(mass_labels[indices[i]], fontsize=fontsize)
        for j in range(2):
            ax[i].annotate('', xy=(arrow_pos, states[i][j + 0]), xytext=(arrow_pos, states[i][j + 1]), \
                        arrowprops=dict(arrowstyle='<->', color='black', mutation_scale=8, shrinkA=shrink, shrinkB=shrink))
        ax[i].text(arrow_pos + 0.05, (states[i][i+0] + states[i][i+1])/2, \
                r'$\Delta m^2_\mathrm{sol}$', va='center', fontsize=fontsize)
        ax[i].text(arrow_pos + 0.05, (states[i][1-i] + states[i][2-i])/2, \
                r'$\Delta m^2_\mathrm{atm}$', va='center', fontsize=fontsize)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].tick_params(axis='both', length=0)
        ax[i].set_xticks([])
        ax[i].set_xlabel(ordering_labels[i], fontsize=fontsize)
    ax[1].legend(frameon=False, handlelength=0.8, loc=[-0.2, 0.4], fontsize=fontsize)

    if save_path:
        fig.savefig(save_path)

    return fig, ax

@plot_in_font
def mass_scale(fermion_masses, params, save_path=None, colors=None, font=None, serif=False):
    """Make a plot showing the absolute scale of neutrino masses compared to
    all other fermions.

    :param fermion_masses: dictionary of fermion masses
    :type fermion_masses: dict
    :param params: dictionary of neutrino oscillation parameters
    :type params: dict
    :param save_path: path where the figure should be saved, defaults to None
    :type save_path: str, optional
    """
    if colors is None:
        all_colors = [to_hex(c) for c in color_sequences['tab10']]
    else:
        all_colors = [to_hex(c) for c in colors]
    nu_colors = [all_colors[3], all_colors[8], all_colors[0]]

    other_colors = []
    for c in all_colors:
        if c not in nu_colors:
            other_colors.append(c)

    fermion_colors = np.array(['#######' for i in range(10)])
    fermion_colors[np.array((2, 5, 8))] = np.array(nu_colors)
    j = 0
    for i in range(len(fermion_colors)):
        if fermion_colors[i] == '#######':
            fermion_colors[i] = other_colors[j]
            j += 1
    fermion_colors[-1], fermion_colors[-3] = fermion_colors[-3], fermion_colors[-1]

    gen_1 = ['up', 'down', 'electron']
    gen_2 = ['charm', 'strange', 'muon']
    gen_3 = ['top', 'bottom', 'tau']

    fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
    ax.set_prop_cycle(color=fermion_colors)
    ax.set_yticks(np.arange(1, 4))
    ax.set_ylabel('Generation')
    ax.set_xlabel('Mass [eV]')
    ax.set_ylim([0.5, 3.5])
    ax.set_xlim([1e-6, 1e12])
    markers = ['^', 'v', 's']
    labels = np.array([['u', 'd', 'e', r'$\nu_1$'], \
                    ['c', 's', r'$\mu$', r'$\nu_2$'], \
                    ['t', 'b', r'$\tau$', r'$\nu_3$']])
    adjustments = np.zeros(labels.shape)
    adjustments[0, 1] = 1.
    adjustments[1, 1] = -1.
    adjustments[1, 2] = 1.
    adjustments[2, 1] = 0.5
    adjustments[2, 2] = -0.5

    mins, maxs = get_mass_ranges(params)

    for i, (mi_min, mi_max) in enumerate(zip(mins, maxs), start=1):
        fermion_masses['nu_' + str(i)] = (mi_min, mi_max)

    for i, gen in enumerate([gen_1, gen_2, gen_3], start=1):
        for j, key in enumerate(gen):
            c, = ax.semilogx(fermion_masses[key], i, marker=markers[j], ls='none', fillstyle='none')
            ax.text(fermion_masses[key]*2.**adjustments[i-1, j], i + 0.15, labels[i-1, j], va='center', color=c.get_color(), style='italic')
        ax.semilogx(fermion_masses['nu_' + str(i)], (i, i), marker='|', ls='-', mew=2, color=c.get_color())
        ax.text(fermion_masses['nu_' + str(i)][-1]*0.2, i + 0.15, labels[i-1, -1], va='center', color=c.get_color(), style='italic')

    ax.fill_betweenx([0, 4], fermion_masses['electron'], fermion_masses['nu_3'][1], color=fermion_colors[-1], hatch='\\\\', alpha=0.15, rasterized=True)

    if save_path:
        fig.savefig(save_path)

    return fig, ax

@plot_in_font
def spinors(creation=False, save_path=None, colors=None, font=None, serif=False):
    """Make a plot showing the chiral composition of neutrino helicity states.

    :param creation: whether to consider state creation, defaults to False
    :type creation: bool, optional
    :param save_path: path where the figure should be saved
    :type save_path: str, optional
    """

    if colors is None:
        colors = color_sequences['tab10']

    # define position and size parameters
    h_supp_factor = 0.1
    width = 1.6
    x_left = 0.4
    x_mid = 2.2
    x_right = 4.0
    y_dirac = 2.3
    y_major = 1.3
    bar_thickness = 0.2
    y_gap = bar_thickness + 0.1
    fontsize = 10
    alpha = 0.5
    hatch = '\\\\\\\\'
    dagger = [r'}$', r'\dagger}$']

    # set up figure
    fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
    ax.axis('off')
    ax.set_xlim([-0.15, 5.85])
    ax.set_ylim([0, 3])
    ax.set_aspect('equal')
    title = ['annihilation', 'creation']
    ax.text(3, 2.94, 'Neutrino state ' + title[creation], ha='center', va='top', fontsize=fontsize + 2)
    color_blue = colors[0]
    color_red = colors[3]

    # add left-handed Dirac spinors
    ax.add_artist(Rectangle((x_left, y_dirac), (1 - h_supp_factor)*width, bar_thickness, \
                            color=color_blue, alpha=alpha, label=r'$\nu^{\!' + dagger[int(creation)]))
    ax.add_artist(CompositePatch(Rectangle, (x_left + width - h_supp_factor*width, y_dirac), h_supp_factor*width, \
                                 bar_thickness, color=color_red, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_left, y_dirac), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_left, y_dirac), (1 - h_supp_factor)*width, bar_thickness, color='k', fill=False))
    ax.add_artist(Rectangle((x_left, y_dirac - y_gap), (1 - h_supp_factor)*width, bar_thickness, \
                            color=color_red, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_left + width - h_supp_factor*width, y_dirac - y_gap), h_supp_factor*width, \
                                 bar_thickness, color=color_blue, alpha=alpha, hatch=hatch, \
                                 label=r'$\nu^{\!' + dagger[not int(creation)]))
    ax.add_artist(Rectangle((x_left, y_dirac - y_gap), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_left, y_dirac - y_gap), (1 - h_supp_factor)*width, bar_thickness, ec='k', fc='none'))

    # add left-handed Majorana spinors
    ax.add_artist(Rectangle((x_left, y_major), (1 - h_supp_factor)*width, bar_thickness, \
                            color=color_blue, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_left + width - h_supp_factor*width, y_major), h_supp_factor*width, \
                                 bar_thickness, color=color_blue, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_left, y_major), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_left, y_major), (1 - h_supp_factor)*width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_left, y_major - y_gap), (1 - h_supp_factor)*width, bar_thickness, \
                            color=color_blue, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_left + width - h_supp_factor*width, y_major - y_gap), h_supp_factor*width, \
                                 bar_thickness, color=color_blue, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_left, y_major - y_gap), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_left, y_major - y_gap), (1 - h_supp_factor)*width, bar_thickness, \
                            ec='k', fc='none'))

    # add v=0 Dirac spinors
    ax.add_artist(Rectangle((x_mid, y_dirac), 0.5*width, bar_thickness, color=color_blue, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_mid + 0.5*width, y_dirac), 0.5*width, bar_thickness, color=color_red, \
                                 alpha=alpha, hatch=hatch, label=r'$\overline{N}^{' + dagger[not int(creation)]))
    ax.add_artist(Rectangle((x_mid, y_dirac), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_mid, y_dirac), 0.5*width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_mid, y_dirac - y_gap), 0.5*width, bar_thickness, color=color_red, \
                            alpha=alpha, label=r'$\overline{N}^{' + dagger[int(creation)]))
    ax.add_artist(CompositePatch(Rectangle, (x_mid + 0.5*width, y_dirac - y_gap), 0.5*width, bar_thickness, \
                                 color=color_blue, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_mid, y_dirac - y_gap), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_mid, y_dirac - y_gap), 0.5*width, bar_thickness, ec='k', fc='none'))

    # add v=0 Majorana spinors
    ax.add_artist(Rectangle((x_mid, y_major), 0.5*width, bar_thickness, color=color_blue, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_mid + 0.5*width, y_major), 0.5*width, bar_thickness, \
                                 color=color_blue, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_mid, y_major), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_mid, y_major), 0.5*width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_mid, y_major - y_gap), 0.5*width, bar_thickness, \
                            color=color_blue, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_mid + 0.5*width, y_major - y_gap), 0.5*width, bar_thickness, \
                                 color=color_blue, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_mid, y_major - y_gap), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_mid, y_major - y_gap), 0.5*width, bar_thickness, ec='k', fc='none'))

    # add right-handed Dirac spinors
    ax.add_artist(Rectangle((x_right, y_dirac), h_supp_factor*width, bar_thickness, \
                            color=color_blue, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_right + h_supp_factor*width, y_dirac), (1 - h_supp_factor)*width, \
                                 bar_thickness, color=color_red, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_right, y_dirac), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_right, y_dirac), h_supp_factor*width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_right, y_dirac - y_gap), h_supp_factor*width, bar_thickness, \
                            color=color_red, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_right + h_supp_factor*width, y_dirac - y_gap), (1 - h_supp_factor)*width, 
                                 bar_thickness, color=color_blue, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_right, y_dirac - y_gap), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_right, y_dirac - y_gap), h_supp_factor*width, bar_thickness, ec='k', fc='none'))

    # add right-handed Majorana spinors
    ax.add_artist(Rectangle((x_right, y_major), h_supp_factor*width, bar_thickness, \
                            color=color_blue, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_right + h_supp_factor*width, y_major), (1 - h_supp_factor)*width, 
                                 bar_thickness, color=color_blue, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_right, y_major), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_right, y_major), h_supp_factor*width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_right, y_major - y_gap), h_supp_factor*width, bar_thickness, \
                            color=color_blue, alpha=alpha))
    ax.add_artist(CompositePatch(Rectangle, (x_right + h_supp_factor*width, y_major - y_gap), (1 - h_supp_factor)*width, \
                                 bar_thickness, color=color_blue, alpha=alpha, hatch=hatch))
    ax.add_artist(Rectangle((x_right, y_major - y_gap), width, bar_thickness, ec='k', fc='none'))
    ax.add_artist(Rectangle((x_right, y_major - y_gap), h_supp_factor*width, bar_thickness, ec='k', fc='none'))

    # add titles and labels
    ax.text(x_left + width/2., (y_dirac + bar_thickness + y_major - y_gap)/2., 'Left-handed', \
            ha='center', va='center', fontsize=fontsize)
    ax.text(x_mid + width/2., (y_dirac + bar_thickness + y_major - y_gap)/2., r'$\leftarrow~helicity~\rightarrow$', \
            ha='center', va='center', fontsize=fontsize)
    ax.text(x_right + width/2., (y_dirac + bar_thickness + y_major - y_gap)/2., 'Right-handed', \
            ha='center', va='center', fontsize=fontsize)
    ax.text(x_left + width/2., y_dirac + 1.1*bar_thickness, r'$v\approx c$', \
            ha='center', va='bottom', fontsize=fontsize)
    ax.text(x_mid + width/2., y_dirac + 1.1*bar_thickness, r'$v\approx 0$', \
            ha='center', va='bottom', fontsize=fontsize)
    ax.text(x_right + width/2., y_dirac + 1.1*bar_thickness, r'$v\approx c$', \
            ha='center', va='bottom', fontsize=fontsize)
    ax.text(x_right + width + 0.3, y_dirac + bar_thickness/2., r'$\nu_e$', \
            ha='right', va='center', fontsize=fontsize)
    ax.text(x_right + width + 0.3, y_dirac - y_gap + bar_thickness/2., r'$\overline{\nu}_e$', \
            ha='right', va='center', fontsize=fontsize)
    ax.text(x_right + width + 0.3, y_major + bar_thickness/2., r'$\nu_e$', \
            ha='right', va='center', fontsize=fontsize)
    ax.text(x_right + width + 0.3, y_major - y_gap + bar_thickness/2., r'$\overline{\nu}_e$', \
            ha='right', va='center', fontsize=fontsize)
    ax.text(x_left - 0.3, y_dirac - (y_gap - bar_thickness)/2., 'Dirac', rotation=90, \
            ha='left', va='center', fontsize=fontsize + 2)
    ax.text(x_left - 0.3, y_major - (y_gap - bar_thickness)/2., 'Majorana', rotation=90, \
            ha='left', va='center', fontsize=fontsize + 2)

    # add legend
    operation = [[r'$\nu_e$-induced IBD', r'$\overline{\nu}_e$-induced IBD'], \
                 [r'$\epsilon/\beta^{\!+}$ decay', r'$\beta^{\!-}$ decay']]

    current = [r'$J_-^\mu=e^{\!\dagger} \overline{\sigma}^{\,\mu} \nu$', \
               r'$J_+^\mu=\nu^{\!\dagger} \overline{\sigma}^{\,\mu} e$']
    ax.text(x_left + 1.1, y_major - y_gap - 0.3, operation[int(creation)][0], fontsize=fontsize)
    ax.text(x_left + 2.4, y_major - y_gap - 0.3, operation[int(creation)][1], fontsize=fontsize)
    ax.text(x_left - 0.2, y_major - y_gap - 0.6, 'Weak currents:', fontsize=fontsize)
    ax.text(x_left + 1.1, y_major - y_gap - 0.6, current[int(creation)], fontsize=fontsize)
    ax.text(x_left + 2.4, y_major - y_gap - 0.6, current[not int(creation)], fontsize=fontsize)
    ax.text(x_left - 0.2, y_major - y_gap - 0.9, 'Field operators:', fontsize=fontsize)
    legend = ax.legend(ncol=4, frameon=False, handlelength=3, handletextpad=0.5, columnspacing=3.3, loc='lower left', \
                       bbox_to_anchor=(0.28, -0.03), fontsize=fontsize, handler_map={CompositePatch: HandlerCompositePatch()})
    legend.set_rasterized(True)

    # save the figure
    if save_path:
        fig.savefig(save_path, dpi=1200)

    return fig, ax

@plot_in_font
def mass_mechanisms(save_path=None, colors=None, font=None, serif=False):
    """Make a diagram illustrating the main neutrino mass generation mechanisms.

    :param save_path: path where the figure should be saved
    :type save_path: str, optional
    :param serif: use serif fonts in the diagram (default is True)
    :type serif: bool
    """
    if colors is None:
        colors = color_sequences['tab10']

    # size of the figure
    x_scale = 6
    y_scale = 4

    # font sizes
    textsize = 12
    symbolsize = 14
    masssize = 14

    # set colors for each group of elements
    scheme = [colors[2], colors[4], colors[3], colors[0], colors[5], colors[1]]
    path1 = scheme[0]
    path2 = scheme[1]
    sm = scheme[2]
    dirac = scheme[3]
    left = scheme[4]
    right = scheme[5]
    alpha = 0.15

    # arrow, line, hatch, and marker parameters
    lw = 1.
    ls = ':'
    arrow_size = lw/10.
    t1_params = {'color':path1, 'width':arrow_size, 'length':3*arrow_size}
    t2_params = {'color':path2, 'width':arrow_size, 'length':3*arrow_size}
    di_params = {'color':dirac, 'width':arrow_size, 'length':3*arrow_size}
    hatch = '/' + '/'*int(18//x_scale)
    ms = 8*lw

    # padding as a fraction of the full width
    x_pad = 0.02
    y_pad = 0.02

    # define the locations of vertices
    x_v = 0.12
    y_v = 0.72
    x_N = 0.21
    y_N = 0.22
    x_M = 0.50
    slope = (y_v - y_N)/(x_v - x_N)
    inter = y_N -slope*x_N
    y_D = 0.5
    x_D = (y_D - y_N)/slope + x_N
    l_h = 0.08
    l_d = 0.13

    # define the box sizes
    w_bg = 0.06*x_scale
    h_bg = 0.10*y_scale
    w_SM = 0.08*x_scale
    h_SM = 0.14*y_scale
    px_D = (x_v - 0.08)*x_scale
    py_D = (y_N - 0.06)*y_scale
    w_D = 0.22*x_scale
    h_D = 0.65*y_scale
    w_t2 = 0.88*x_scale
    h_t2 = 0.28*y_scale
    py_t2 = 0.63*y_scale
    w_t1 = 0.26*x_scale
    h_t1 = 0.75*y_scale
    h2_t1 = 0.25*h_t1
    px_t1 = 0.02*x_scale
    py_t1 = 0.10*y_scale

    # create a new figure
    fig, ax = plt.subplots(figsize=(x_scale, y_scale), layout='constrained')
    diagram = feynman.Diagram(ax)

    # define all the vertices
    v1 = diagram.vertex(xy=(x_v*x_scale, y_v*y_scale), marker='')
    v2 = diagram.vertex(xy=(x_D*x_scale, y_D*y_scale), marker='.', ms=0.5*ms, color=dirac)
    v6 = diagram.vertex(xy=((1 - x_D)*x_scale, y_D*y_scale), marker='.', ms=0.5*ms, color=dirac)
    v9 = diagram.vertex(xy=((x_D - l_h)*x_scale, y_D*y_scale), marker='x', ms=ms, color=dirac, mew=lw)
    v10 = diagram.vertex(xy=((1 - x_D + l_h)*x_scale, y_D*y_scale), marker='x', ms=ms, color=dirac, mew=lw)
    v3 = diagram.vertex(xy=(x_N*x_scale, y_N*y_scale), marker='')
    v4 = diagram.vertex(xy=(x_M*x_scale, y_N*y_scale), marker='x', ms=ms, color=path1, mew=lw)
    v5 = diagram.vertex(xy=((1 - x_N)*x_scale, y_N*y_scale), marker='')
    v7 = diagram.vertex(xy=((1 - x_v)*x_scale, y_v*y_scale), marker='')
    v8 = diagram.vertex(xy=(x_M*x_scale, y_v*y_scale), marker='.', ms=0.5*ms, color=path2, mew=lw)
    v11 = diagram.vertex(xy=(x_M*x_scale, (y_v + l_d)*y_scale), marker='x', ms=ms, color=path2, mew=lw)

    # add lines between vertices
    l1 = diagram.line(v1, v2, lw=lw, color=dirac, arrow_param=di_params)
    l2 = diagram.line(v3, v2, lw=lw, color=dirac, arrow_param=di_params)
    l5 = diagram.line(v5, v6, lw=lw, color=dirac, arrow_param=di_params)
    l6 = diagram.line(v7, v6, lw=lw, color=dirac, arrow_param=di_params)
    l9 = diagram.line(v2, v9, lw=lw, ls='--', color=dirac, arrow=False)
    l10 = diagram.line(v6, v10, lw=lw, ls='--', color=dirac, arrow=False)
    l3 = diagram.line(v4, v3, lw=lw, color=path1, arrow_param=t1_params)
    l4 = diagram.line(v4, v5, lw=lw, color=path1, arrow_param=t1_params)
    l7 = diagram.line(v7, v8, lw=lw, color=path2, arrow_param=t2_params)
    l8 = diagram.line(v1, v8, lw=lw, color=path2, arrow_param=t2_params)
    l11 = diagram.line(v8, v11, lw=lw, ls='--', color=path2, arrow=False)

    # background behind labels
    bg1 = Rectangle((x_v*x_scale - w_bg/2., y_v*y_scale - h_bg/2.), w_bg, h_bg, fc='white', ec='none', zorder=99)
    bg2 = Rectangle(((1 - x_v)*x_scale - w_bg/2., y_v*y_scale - h_bg/2.), w_bg, h_bg, fc='white', ec='none', zorder=99)
    ax.add_patch(bg1)
    ax.add_patch(bg2)

    # box defining interactions in minimal SM
    r1 = Rectangle((x_v*x_scale - w_SM/2., y_v*y_scale - h_SM/2.), w_SM, h_SM, fc='none', \
                ec=sm, lw=lw, ls=ls, zorder=999)
    r2 = Rectangle(((1 - x_v)*x_scale - w_SM/2., y_v*y_scale - h_SM/2.), w_SM, h_SM, fc='none', \
                ec=sm, lw=lw, ls=ls, zorder=999)
    ax.add_patch(r1)
    ax.add_patch(r2)

    # box defining interactions for pure Dirac neutrinos
    r3 = Rectangle((px_D, py_D), w_D, h_D, ec=dirac, fc='none', lw=lw, ls=ls, zorder=999)
    r4 = Rectangle((x_scale - px_D - w_D, py_D), w_D, h_D, ec=dirac, fc='none', lw=lw, ls=ls, zorder=999)
    ax.add_patch(r3)
    ax.add_patch(r4)

    # regions defining chirality and particle/antiparticle state
    r5 = Rectangle((-x_pad*x_scale, -y_pad*y_scale), (0.5 + x_pad)*x_scale, (0.5 + y_pad)*y_scale, \
                ls='none', color=right, alpha=alpha, lw=lw, zorder=100)
    r7 = Rectangle((-x_pad*x_scale, 0.50*y_scale), (0.5 + x_pad)*x_scale, (0.5 + y_pad)*y_scale, \
                ls='none', color=left, alpha=alpha, lw=lw, zorder=100)
    with plt.rc_context({'hatch.linewidth': lw}):
        r6 = CompositePatch(Rectangle, (0.50*x_scale, -y_pad*y_scale), (0.5 + x_pad)*x_scale, (0.5 + y_pad)*y_scale, \
                            ls='none', color=left, hatch=hatch, alpha=alpha, lw=lw, zorder=100)
        r8 = CompositePatch(Rectangle, (0.5*x_scale, 0.50*y_scale), (0.5 + x_pad)*x_scale, (0.5 + y_pad)*y_scale, \
                            ls='none', color=right, hatch=hatch, alpha=alpha, lw=lw, zorder=100)
    ax.add_artist(r5)
    ax.add_artist(r6)
    ax.add_artist(r7)
    ax.add_artist(r8)

    # box defining interactions for Type-II seesaw mechanism
    r9 = Rectangle(((x_scale - w_t2)/2., py_t2), w_t2, h_t2, ec=path2, fc='none', lw=lw, ls=ls, zorder=999)
    ax.add_artist(r9)

    # region defining interactions for Type-I seesaw mechanism
    b1 = Line2D((px_t1, px_t1), (py_t1, py_t1 + h_t1), color=path1, lw=lw, \
                ls=ls, zorder=999)
    b2 = Line2D((x_scale - px_t1, x_scale - px_t1), (py_t1, py_t1 + h_t1), \
                color=path1, lw=lw, ls=ls, zorder=999)
    b3 = Line2D((px_t1, x_scale - px_t1), (py_t1, py_t1), color=path1, lw=lw, \
                ls=ls, zorder=999)
    b4 = Line2D((px_t1, px_t1 + w_t1), (py_t1 + h_t1, py_t1 + h_t1), \
                color=path1, lw=lw, ls=ls, zorder=999)
    b5 = Line2D((x_scale - px_t1 - w_t1, x_scale - px_t1), (py_t1 + h_t1, py_t1 + h_t1), \
                color=path1, lw=lw, ls=ls, zorder=999)
    b6 = Line2D((px_t1 + w_t1, px_t1 + w_t1), (py_t1 + h_t1, py_t1 + h2_t1), \
                color=path1, lw=lw, ls=ls, zorder=999)
    b7 = Line2D((x_scale - px_t1 - w_t1, x_scale - px_t1 - w_t1), (py_t1 + h_t1, py_t1 + h2_t1), \
                color=path1, lw=lw, ls=ls, zorder=999)
    b8 = Line2D((px_t1 + w_t1, x_scale - px_t1 - w_t1), (py_t1 + h2_t1, py_t1 + h2_t1), \
                color=path1, lw=lw, ls=ls, zorder=999)
    ax.add_line(b1)
    ax.add_line(b2)
    ax.add_line(b3)
    ax.add_line(b4)
    ax.add_line(b5)
    ax.add_line(b6)
    ax.add_line(b7)
    ax.add_line(b8)

    # labels for fermion lines
    diagram.text(x_v*x_scale, y_v*y_scale, r'$\nu$', fontsize=symbolsize, zorder=999)
    diagram.text((1 - x_v)*x_scale, y_v*y_scale, r'$\nu$', fontsize=symbolsize, zorder=999)
    diagram.text(x_D*x_scale, y_N*y_scale, r'$\overline{N}$', fontsize=symbolsize, zorder=999)
    diagram.text((1 - x_D)*x_scale, y_N*y_scale, r'$\overline{N}$', fontsize=symbolsize, zorder=999)
    diagram.text((x_D - l_h + 0.02)*x_scale, (y_D - 0.07)*y_scale, r'$\left<\Phi\right>$', \
                color=dirac, fontsize=symbolsize, zorder=999)
    diagram.text((1 - x_D + l_h - 0.02)*x_scale, (y_D - 0.07)*y_scale, r'$\left<\Phi\right>$', \
                color=dirac, fontsize=symbolsize, zorder=999)
    diagram.text(0.45*x_scale, (y_v + l_d - 0.03)*y_scale, r'$\left<\Delta\right>$', \
                color=path2, fontsize=symbolsize, zorder=999)

    # mass generation mechanism labels
    diagram.text(0.5*x_scale, 0.55*y_scale, 'Minimal Standard Model', color=sm, fontsize=textsize, zorder=999)
    diagram.text(0.5*x_scale, 0.45*y_scale, 'Pure Dirac', fontsize=textsize, color=dirac, zorder=999)
    diagram.text(0.5*x_scale, 0.95*y_scale, 'Type-II seesaw', color=path2, fontsize=textsize, zorder=999)
    diagram.text(0.5*x_scale, 0.35*y_scale, 'Type-I seesaw', color=path1, fontsize=textsize, zorder=999)

    # chirality labels
    diagram.text(0.13*x_scale, 0.95*y_scale, 'Left-handed', fontsize=textsize, color=left, zorder=999)
    diagram.text(0.87*x_scale, 0.95*y_scale, 'Right-handed', fontsize=textsize, color=right, zorder=999)
    diagram.text(0.11*x_scale, 0.04*y_scale, '$L=+1$', fontsize=textsize, color='k', zorder=999)
    diagram.text(0.89*x_scale, 0.04*y_scale, '$L=-1$', fontsize=textsize, color='k', zorder=999)
    diagram.text(0.50*x_scale, 0.04*y_scale, r'$Time$  $\longrightarrow$', fontsize=textsize, zorder=999)

    # labels for mass insertions
    diagram.text((x_D + 0.04)*x_scale, y_D*y_scale, r'$y_\nu$', fontsize=masssize, color=dirac, zorder=999)
    diagram.text((1 - x_D - 0.04)*x_scale, y_D*y_scale, r'$y_\nu$', fontsize=masssize, color=dirac, zorder=999)
    diagram.text(0.50*x_scale, 0.15*y_scale, r'$m_\mathrm{M}$', fontsize=masssize, color=path1, zorder=999)
    diagram.text(0.50*x_scale, 0.67*y_scale, r'$y_\mathrm{L}$', fontsize=masssize, color=path2, zorder=999)

    # plot everything and set the axes
    diagram.plot()
    ax.set_axis_off()
    ax.set_xlim([-x_pad*x_scale, x_scale*(1 + x_pad)])
    ax.set_ylim([-y_pad*y_scale, y_scale*(1 + y_pad)])
    ax.set_aspect('equal')

    if save_path:
        fig.savefig(save_path, pad_inches=-0.1)

    return fig, ax


@plot_in_font
def decay_chain(chain, save_path=None, colors=None, font=None, serif=None):
    """Plot a decay chain given the decay data series.

    :param chain: the decay data to plot
    :type chain: dict
    :param save_path: path where the figure should be saved
    :type save_path: str, optional
    """
    vertices = np.array(((0, 0), (0, 1), (1/np.sqrt(2), 1 + 1/np.sqrt(2)), \
                     (1 + 1/np.sqrt(2), 1 + 1/np.sqrt(2)), (1 + 2/np.sqrt(2), 1), \
                     (1 + 2/np.sqrt(2), 0), (1 + 1/np.sqrt(2), -1/np.sqrt(2)), \
                     (1/np.sqrt(2), -1/np.sqrt(2)), (0, 0))) - np.array((0.5 + 1/np.sqrt(2), 0.5))
    scale = 0.2
    z_spacing = 2.8
    font_base = 12
    lw_base = 10
    hl_offset = np.array((0, -scale*0.7))
    sym_offset = np.array((0, scale*0.05))
    alpha_offset = np.array((0, -0.5 - 1/np.sqrt(2)))*scale
    beta_offset = np.array((0.5 + 0.5/np.sqrt(2), 0.5 + 0.5/np.sqrt(2)))*scale
    family_keys = np.array(['act', 'alk-e', 'nob', 'chalc', 'hal', 'post'])

    if colors is None:
        colors = ['#8ab0cb', '#ffd6ad', '#a7af8c', '#cccaff', '#e5a8cd', '#b9a7a8']

    def draw_radioisotope(axis, symbol, A, Z, halflife, alpha, beta, family, br):
        position = np.array(((Z - A/2)*z_spacing*scale, Z*z_spacing*scale))
        color = to_rgba(colors[np.argwhere(family_keys==family)[0][0]], alpha=0.5)
        p = Polygon(position + scale*vertices, fc=color, ec='k', lw=lw_base*scale/z_spacing)
        axis.add_artist(p)
        axis.text(*(position + sym_offset), '$^{{{}}}_{{{}}}${}'.format(A, '~~' + str(Z), symbol), ha='center', va='center', fontsize=10*font_base*scale/z_spacing)
        axis.text(*(position + hl_offset), halflife, ha='center', va='center', fontsize=6*font_base*scale/z_spacing)
        if alpha:
            ls = '-'
            if br <= 1e-2:
                ls = ':'
            axis.annotate("", xytext=position + alpha_offset, \
                          xy=position - alpha_offset - np.array((0, 2*z_spacing*scale)), \
                          arrowprops=dict(arrowstyle="-|>", mutation_scale=15*lw_base*scale/z_spacing, \
                                          shrinkA=0, shrinkB=0, color='k', lw=lw_base*scale/z_spacing, ls=ls))
            axis.text(*(position + alpha_offset + np.array((0.15*scale*z_spacing, -0.5*scale*z_spacing))), \
                      r'$\alpha$', ha='center', va='center', fontsize=8*font_base*scale/z_spacing)
        if beta:
            ls = '-'
            if br >= 1 - 1e-2:
                ls = ':'
            axis.annotate("", xytext=position + beta_offset, \
                          xy=position - beta_offset + np.array((z_spacing*scale, z_spacing*scale)), \
                          arrowprops=dict(arrowstyle="-|>", mutation_scale=15*lw_base*scale/z_spacing, \
                                          shrinkA=0, shrinkB=0, color='k', lw=lw_base*scale/z_spacing, ls=ls))
            axis.text(*(position + beta_offset + np.array((0.4*scale*z_spacing, 0.*scale*z_spacing))), \
                      r'$\beta^{-}$', ha='center', va='center', fontsize=8*font_base*scale/z_spacing)

    fig, ax = plt.subplots(figsize=(4, 6), layout='constrained')

    x_vals = []
    for d in chain:
        x_vals.append(d['Z'] - d['A']/2)
    x_vals = np.array(x_vals)

    Z_min = np.amin([d['Z'] for d in chain])
    Z_max = np.amax([d['Z'] for d in chain])

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    ax.set_ylim([(Z_min - 0.5)*z_spacing*scale, (Z_max + 0.5)*z_spacing*scale])
    ax.set_xlim([np.amin(x_vals - 0.5)*z_spacing*scale, np.amax(x_vals + 0.5)*z_spacing*scale])

    for d in chain:
        draw_radioisotope(ax, d['symbol'], d['A'], d['Z'], d['halflife'], d['alpha'], d['beta'], d['family'], d['br'])

    if save_path:
        fig.savefig(save_path)

    return fig, ax

