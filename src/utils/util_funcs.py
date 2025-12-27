import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.font_manager as fm
import os
import glob


def basis_change(sigma, dm2_21, dm2_23):
    """Finds the 3 individual neutrino masses given the sum of masses and two
    mass-squared differences. Determines hierarchy based on the sign of dm2_23.
    """
    if dm2_23 > 0:
        def residual(m1):
            try:
                m2 = np.sqrt(m1**2 + dm2_21)
                m3 = np.sqrt(m2**2 + dm2_23)
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
        m3 = np.sqrt(m2**2 + dm2_23)
        return np.array([m1, m2, m3]), True

    elif dm2_23 < 0:
        def residual(m3):
            try:
                m2 = np.sqrt(m3**2 - dm2_23)
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
        m2 = np.sqrt(m3**2 - dm2_23)
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


def create_text_figure(text, save_path=None, fontsize=24, color='k', facecolor='none', \
                       padding=0.25, dpi=None):
    """Creates a figure with the given text in matplotlib. Used to make
    equations in latex for slides.

    :param text: 
    :type text: _type_
    :param save_path: _description_, defaults to None
    :type save_path: _type_, optional
    :param fontsize: _description_, defaults to 24
    :type fontsize: int, optional
    :param color: _description_, defaults to 'k'
    :type color: str, optional
    :param facecolor: _description_, defaults to 'none'
    :type facecolor: str, optional
    :param dpi: _description_, defaults to None
    :type dpi: _type_, optional
    """
    fig, ax = plt.subplots(facecolor=facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    if facecolor == 'none':
        ax.patch.set_alpha(0)

    # create text at origin
    text = ax.text(0, 0, text, fontsize=fontsize, ha='center', va='center', color=color)

    # draw to get correct bounds
    try:
        fig.canvas.draw()
    except RuntimeError as e:
        plt.close(fig)
        print('Error: check your latex!')
        print(e)
        return
    bbox = text.get_window_extent()
    bbox_fig = bbox.transformed(fig.dpi_scale_trans.inverted())
    bbox_data = bbox.transformed(ax.transData.inverted())

    # add padding
    width = bbox_fig.width + 2 * padding
    height = bbox_fig.height + 2 * padding

    fig.set_size_inches(width, height)

    # convert padding from inches to data coordinates
    padding_data = padding * (bbox_data.width / bbox_fig.width)

    # set axis limits with padding
    half_width = (bbox_data.x1 - bbox_data.x0)/2 + padding_data
    half_height = (bbox_data.y1 - bbox_data.y0)/2 + padding_data

    ax.set_xlim([-half_width, half_width])
    ax.set_ylim([-half_height, half_height])

    # set margins to 0 since we're handling padding in data coordinates
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, \
                    transparent=[False, True][int(facecolor=='none')], dpi=dpi)
        print('Success: figure saved to ' + save_path)


class CompositePatch(mpatches.Patch):
    """Class to create patches with both hatching and transparency
    that will save correctly to vector formats.
    """
    def __init__(self, patch_class, *args,
                 color=None, facecolor=None, fc=None, edgecolor=None,
                 ec=None, alpha=None, hatch=None, label=None,
                 **kwargs):
        """The first argument should be a matplotlib.mpatches.Patch class,
        then the remaining arguments are the usual arguments to Patch.
        """
        
        zorder = kwargs.pop("zorder", None)
        if fc is None:
            fc = facecolor if facecolor is not None else color
        if ec is None:
            ec = edgecolor if edgecolor is not None else color

        self.patch_class = patch_class
        self.args = args
        self.kwargs = kwargs
        self.alpha = alpha
        self.hatch = hatch

        # create the two patches
        self.fill_patch = patch_class(*args, fc=fc, ec='none',
                                      alpha=alpha, zorder=zorder, **kwargs)
        self.hatch_patch = patch_class(*args, fc='none', ec=ec, hatch=hatch,
                                       alpha=alpha, zorder=zorder, **kwargs)
        self._children = [self.fill_patch, self.hatch_patch]

        super().__init__(label=label)

        if zorder is not None:
            self.set_zorder(zorder)

    def draw(self, renderer):
        for child in self._children:
            child.axes = self.axes
            child.set_transform(self.get_transform())
            child.draw(renderer)

    def set_zorder(self, z):
        for c in self._children:
            c.set_zorder(z)
        super().set_zorder(z)

    def get_zorder(self):
        return self._children[0].get_zorder()

    def set_clip_path(self, clip_path):
        for c in self._children:
            c.set_clip_path(clip_path)
        super().set_clip_path(clip_path)

    def get_children(self):
        return self._children

    def contains_point(self, point, radius=None):
        return any(c.contains_point(point, radius) for c in self._children)

    def get_bbox(self):
        return self.fill_patch.get_bbox()


class HandlerCompositePatch(HandlerPatch):
    """Class to create legend artists for patches created using the
    custom CompositePatch class.
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        patch = orig_handle.patch_class(*orig_handle.args,
                                        fc=orig_handle.fill_patch.get_facecolor(),
                                        ec=orig_handle.hatch_patch.get_edgecolor(),
                                        hatch=orig_handle.hatch,
                                        alpha=orig_handle.alpha,
                                        **orig_handle.kwargs)
        patch.set_transform(trans)
        patch.set_x(xdescent)
        patch.set_y(ydescent)
        patch.set_width(width)
        patch.set_height(height)
        return [patch]
    

def set_fonts(font_name, font_path, serif=False):
    """Set the font used by matplotlib.

    :param font_name: Name of the font
    :type font_name: str
    :param font_path: Path to the folder containing the .ttf files
    :type font_path: str
    """
    font_family = 'serif' if serif else 'sans-serif'
    fonts_dir = os.path.abspath(font_path)
    font_variants = glob.glob(fonts_dir + '/' + '*.ttf')

    for font in font_variants:
        font_path = os.path.join(fonts_dir, font)
        fm.fontManager.addfont(font_path)

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.' + font_family] = font_name