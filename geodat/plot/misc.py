from itertools import cycle
import pylab
import numpy
from matplotlib.patches import PathPatch


def contour_to_hatched_patches(cntrset, hatch_colors, hatch_patterns,
                               remove_contour=True):
    ''' Add hatch patches to contour plots

    Args:
        cntrset (matplotlib.contour.ContourSet): contour map
        hatch_colors (iterable): hatch colors
        hatch_patterns (iterable): hatch_patterns
        remove_contour (boolean): if True, linewidth of patches is 0
    '''
    patches_list = []
    for pathcollection in cntrset.collections:
        patches_list.append([PathPatch(p1)
                             for p1 in  pathcollection.get_paths()])
        if remove_contour:
            pathcollection.remove()

    for patches, hc, hp  in zip(patches_list,
                                cycle(hatch_colors), cycle(hatch_patterns)):
        for p in patches:
            p.set_fc("none")
            p.set_ec("k")
            if remove_contour: p.set_lw(0)
            p.set_hatch(hp)
            cntrset.ax.add_patch(p)


def reorderlegend(orderedlegend=None):
    ''' Reorder in the line using `orderedlegend`

    Arg: 
        orderedlegend (list of str): if None, the current legends will be sorted

    Returns:
        matplotlib.legend.Legend
    '''
    l = pylab.legend()
    current_legend = [ v.get_text() for v in l.get_texts() ]
    if orderedlegend is None:
        orderedlegend = sorted(current_legend)
    assert sorted(current_legend) == sorted(orderedlegend)
    ax = pylab.gca()
    ax.lines = [ ax.lines[current_legend.index(t)] for t in orderedlegend ]
    l = pylab.legend()
    return l


def bar_plot(xlabel=None,y=None,data=None,**kwargs):
    ''' Shortcut for making multiple bar_plot with xlabel

    Args:
       xlabel (list of str): x axis labels
       y (numpy array): values for bar plot
       data (dict): keys as xlabel, values as y

    Other keyword arguments go to set_xticklabels

    Returns: None
    '''
    if data is not None and type(data) is dict:
        xlabel = data.keys()
        y = data.values()

    if y is None:
        raise ValueError("y is not provided manually or from data")

    xloc = numpy.arange(len(y))+1.
    pylab.bar(xloc,y,label='_nolegend_',align='center')
    pylab.gca().set_xticks(xloc)

    if xlabel is not None:
        pylab.gca().xaxis.set_ticks_position('none')
        pylab.gca().set_xticklabels(xlabel,**kwargs)


def tick_formatter(ax,scilimits=(-3,4),
                   formatter=pylab.ScalarFormatter,
                   offset_loc_pads=None):
    fmt = formatter()
    fmt.set_powerlimits(scilimits)
    fmt.set_scientific(True)
    fmt._useMathText = True

    def colorbar_set_formatter(fmt):
        ax.formatter = fmt
        ax.update_ticks()

    setters = { 'x': pylab.gca().get_xaxis().set_major_formatter,
                'y': pylab.gca().get_yaxis().set_major_formatter,
                'cbar': colorbar_set_formatter}

    if type(ax) is str:
        set_formatter = setters[ax.lower()]
    elif isinstance(ax,pylab.matplotlib.colorbar.Colorbar):
        set_formatter = setters['cbar']
    else:
        set_formatter = ax.set_major_formatter

    set_formatter(fmt)
    pylab.draw()
    # offset the location of scientific notation
    if offset_loc_pads is not None:
        if not hasattr(offset_loc_pads,"__len__"):
            raise TypeError("offset_loc_pads has no length")
        if len(offset_loc_pads) != 2:
            raise ValueError(("offset_loc_pads need have a length of 2"
                              " for x and y positions of the offsetText"))
        x,y = fmt.axis.offsetText.get_position()
        x += offset_loc_pads[0]* pylab.gcf().dpi / 72.0
        y += offset_loc_pads[1]* pylab.gcf().dpi / 72.0
        fmt.axis.offsetText.set_position((x,y))
        pylab.draw()
    return fmt


def colorbar(cs=None,vmin=None,vmax=None, **kwargs):
    ''' Create a colorbar with the vmin,vmax if vmin and vmax are
    user-defined, rescale the filled contour as well.  '''
    fig = pylab.gcf()
    if cs is not None:
        if vmin is not None and vmax is not None:
            cs.set_clim(vmin,vmax)
        if vmin is None: vmin = cs.get_clim()[0]
        if vmax is None: vmax = cs.get_clim()[1]
        mappable = pylab.cm.ScalarMappable(
            cmap=cs.get_cmap(),
            norm=pylab.normalize(vmin=vmin,vmax=vmax))

        mappable.set_array([])

        cbar = fig.colorbar(mappable,**kwargs)
    else:
        cbar = pylab.colorbar(**kwargs)

    tick_formatter(cbar)
    return cbar
