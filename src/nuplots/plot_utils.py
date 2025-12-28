import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.font_manager as fm
import os
import glob


def create_text_figure(text, save_path=None, fontsize=24, color='k', facecolor='none', \
                       padding=0.25, dpi=None):
    """Creates a figure with the given text in matplotlib. Used to make
    equations in latex for slides.

    :param text: text to plot
    :type text: str
    :param save_path: path where the figure should be saved, defaults to None
    :type save_path: str, optional
    :param fontsize: font size, defaults to 24
    :type fontsize: int, optional
    :param color: text color, defaults to 'k'
    :type color: str, optional
    :param facecolor: text facecolor, defaults to 'none'
    :type facecolor: str, optional
    :param dpi: figure dpi, defaults to None
    :type dpi: int, optional
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
    mathfont = 'cm' if serif else 'dejavusans'

    if font_name not in fm.fontManager.get_font_names():
        fonts_dir = os.path.abspath(font_path)
        font_variants = glob.glob(fonts_dir + '/' + '*.ttf')

        for font in font_variants:
            font_path = os.path.join(fonts_dir, font)
            fm.fontManager.addfont(font_path)

    return {'text.usetex': False, \
            'font.family': font_family, \
            'font.' + font_family: font_name, \
            'mathtext.fontset': mathfont}