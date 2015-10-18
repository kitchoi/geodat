import pylab

def contourf(cses,orientation='horizontal',
             vmin=None,vmax=None,sym=False,
             adjust_add_axes_function=None,
             add_colorbar=True,bottom=0.25,
             right=0.75,
             **kwargs):
    ''' Assumed horizonal colorbar at the moment.
    Will add options for other orientation
    sym : whether the colorbar scale has to be symmetric about zero
    '''
    import plot
    fig = pylab.gcf()
    clims = [ cs.get_clim() for cs in cses ]
    vmins = [ clim[0] for clim in clims ]
    vmaxs = [ clim[1] for clim in clims ]
    if vmin is None: vmin = min(vmins)
    if vmax is None: vmax = max(vmaxs)
    if sym and vmin < 0:
        vmin = max(abs(vmin),abs(vmax))*-1
        vmax = vmin*-1
    for cs in cses:
        cs.set_clim(vmin,vmax)
    
    if add_colorbar:
        if orientation == 'horizontal':
            pylab.subplots_adjust(bottom=bottom)
            cbar_ax = fig.add_axes([0.1,bottom-0.1,0.85,0.05])
        elif orientation == 'vertical':
            pylab.subplots_adjust(right=right)
            cbar_ax = fig.add_axes([right+0.1,0.08,0.03,0.8])
        else:
            raise Exception("Unexpected orientation:'horizontal'/'vertical'")
        cbar = plot.colorbar(cses[-1],
                             orientation=orientation,cax=cbar_ax,**kwargs)
        if kwargs.has_key('label'):
            cbar.set_label(kwargs['label'])

    else:
        cbar = None
    
    return cbar

def share_axis(axes,option,axis):
    ''' Pre-condition: fig,axes = pylab.subplots 
    has already been called.  axes as the input.
    
    Input:
    axes - list of axes
    option - "row", "col" or "all"
    axis - "x" or "y"
    '''
    import numpy
    if axis == "y":
        getaxis = lambda ax: ax.get_shared_y_axes()
    elif axis == "x":
        getaxis = lambda ax : ax.get_shared_x_axes()
    else:
        raise Exception("Unexpected axis")
    
    # if there is only one row or one col:
    if len(axes.shape) == 1:
        if axes[-1].is_first_row(): # one row
            axes = axes[numpy.newaxis,]
        else: # one col
            axes = axes[...,numpy.newaxis]
    
    def share_row(axes):
        for row in axes:
            ax1 = row[0]
            for ax in row[1:]:
                getaxis(ax).join(ax,ax1)
    
    def share_col(axes):
        ncols = len(axes[0])
        ax_cols = [ ax for ax in axes[0] ]
        for row in axes[1:]:
            for iax in range(ncols):
                getaxis(row[iax]).join(row[iax],ax_cols[iax])

    if option == "row":
        share_row(axes)
    elif option == "col":
        share_col(axes)
    elif option == "all":
        share_row(axes)
        share_col(axes)
    else:
        raise Exception("Unexpected option")
    
    return axes


def suplabel(axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    fig = pylab.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None: 
        label_prop = dict()
    pylab.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,
               **label_prop)



def suplabel2(axis,label,label_prop=None):
    ''' Add super ylabel or xlabel to the figure
    This function uses fig.add_subplot(111) to create a colorless Axes
    The axes may reappear when exporting
    axis       - string: "x" or "y"
    label      - string
    label_prop - dictionary; keyword arguments for set_label
    '''
    fig = pylab.gcf()
    if len(fig.axes) > 1:
        ax = fig.add_subplot(111,axisbg='none',zorder=-1)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', 
                       bottom='off', left='off', right='off')
    else:
        # there is only one axes
        ax = pylab.gca()
    if axis.lower() == "y":
        label_fun = ax.set_ylabel
    elif axis.lower() == 'x':
        label_fun = ax.set_xlabel
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None: label_prop = {}
    label_fun(label,**label_prop)
    pylab.draw()


def add_subplot_name(offsets,subplot_names="abcdefghijklmnopq"):
    if len(offsets) != 2:
        raise Exception("Offsets for x and y coordinates should be provided")
    for iax,ax in enumerate(pylab.gcf().axes):
        pylab.sca(ax)
        pylab.text(offsets[0],1.+offsets[1],"({})".format(subplot_names[iax]),
                   transform=pylab.gca().transAxes)


