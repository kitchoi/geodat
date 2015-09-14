import pylab
import template

def text(string,x=1,y=0.,halign='right',
         valign='bottom',**kwargs):
    ''' Shortcut to add text to the bottom right corner of the figure.
    Can change location.  '''
    ax = pylab.gca()
    pylab.text(x,y,string,horizontalalignment=halign,
               verticalalignment=valign,
               transform=ax.transAxes,**kwargs)



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


def reorderlegend(orderedlegend=None):
    ''' Reorder in the line using the legend text Input: A list of
    legend text strings, if None, the current legend string will be
    sorted '''
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
    import numpy
    if data is not None:
        if type(data) is dict:
            xlabel = data.keys()
            y = data.values()
    xloc = numpy.arange(len(y))+1.
    pylab.bar(xloc,y,label='_nolegend_',align='center')
    pylab.gca().set_xticks(xloc)
    if xlabel is not None:
        pylab.gca().xaxis.set_ticks_position('none')
        pylab.gca().set_xticklabels(xlabel,**kwargs)
