import pylab
import numpy
from matplotlib.patches import PathPatch

def contour_to_hatched_patches(cntrset, hatch_colors, hatch_patterns,
                               remove_contour=True):
    from itertools import cycle

    patches_list = []
    for pathcollection in cntrset.collections:
        patches_list.append([PathPatch(p1) for p1 in  pathcollection.get_paths()])
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


def grouped_barboxplot(plot_option,group,keys1=None,keys2=None,
                       group_names=None,bar_names=None,bar_colors=None,
                       boxprops=None,boxplot_kwargs=None,legend_props=None,
                       hatches=None,xticklabels_kwargs=None,
                       bar_marks_funcs=[lambda ls: numpy.median(ls)-numpy.percentile(ls,25.),
                                        lambda ls: numpy.median(ls),
                                        lambda ls: numpy.percentile(ls,75.)-numpy.median(ls)],
                       ecolors="k"):
    plot_option = plot_option.lower()
    assert plot_option == 'boxplot' or plot_option == 'bar'
    # Make sure group is a dictionary of dictionary
    assert isinstance(group,dict)
    assert all([ isinstance(bars,dict) for bars in group.values() ])
    # Make sure group is not empty
    assert len(group) > 0
    assert len(group.values()) > 0
    
    if keys1 is None:
        keys1 = group.keys()
    if keys2 is None:
        keys2 = group.values()[0].keys()
    if group_names is None:
        group_names = dict(zip(keys1,keys1))
    else:
        assert isinstance(group_names,dict)
        assert all([ name in group_names.keys() for name in keys1 ])
    if bar_names:
        assert isinstance(bar_names,dict)
        assert all([ name in bar_names.keys() for name in keys2 ])
    if bar_colors is not None:
        assert len(bar_colors) == len(keys2)
    
    ind = numpy.arange(len(keys1))  # the x locations for the groups
    nbars = len(keys2)
    width = 1./(nbars+1)       # the width of the bars
    if bar_names:
        bar_labels = [ bar_names[bar_name] for bar_name in keys2 ]
    else:
        bar_labels = keys2
    
    if plot_option == 'boxplot':
        for igp,key1 in enumerate(keys1):
            for ibar,key2 in enumerate(keys2):
                value = group[key1][key2]
                # If bar colors is given, define it
                if boxprops:
                    box_kwargs = { key: value[ibar] for key,value in boxprops.items() }
                else:
                    box_kwargs = {}
                if boxplot_kwargs is None:
                    boxplot_kwargs = {}
                # if the value is a list or tuple, find the mean and standard deviation
                if isinstance(value,list) or isinstance(value,tuple):
                    boxplots = pylab.boxplot(value,positions=[ind[igp]+ibar*width,],
                                             patch_artist=True,**boxplot_kwargs)
                    boxplots['boxes'][0].set(**box_kwargs)
                else:
                    pylab.scatter(ind[igp]+ibar*width, value,)
        # Add legend
        # Default colors
        boxcolors = ['b',]*len(keys2)
        # Customed colors
        if boxprops:
            if 'color' in boxprops:
                boxcolors = [ boxprops['color'][ibar] for ibar in range(len(keys2)) ]
        rects = []
        for ibar, bar_label in enumerate(bar_labels):
            rects.append(pylab.gca().add_patch(pylab.Rectangle((0.1,0.2),0,0,color=boxcolors[ibar],label=bar_label)))
        #for rect in rects:
        #    rect.set_visible(False)
        if legend_props is not None:
            pylab.gca().legend(**legend_props)
    elif plot_option=='bar':
        rects = []
        for igp,key1 in enumerate(keys1):
            for ibar,key2 in enumerate(keys2):
                value = group[key1][key2]
                # If bar colors is given, define it
                if bar_colors:
                    bar_kwargs = {'color':bar_colors[ibar]}
                else:
                    bar_kwargs = {}
                if hatches:
                    bar_kwargs['hatch'] = hatches[ibar]
                if ecolors:
                    if hasattr(ecolors,"__iter__"):
                        ecolor = ecolors[ibar]
                    else:
                        ecolor= ecolors
                # if the value is a list or tuple, find the mean and standard deviation
                if isinstance(value,list) or isinstance(value,tuple):
                    value_std = [bar_marks_funcs[0](value),bar_marks_funcs[2](value)]
                    value_mean = bar_marks_funcs[1](value)
                    rects.append(
                        pylab.bar(ind[igp]+ibar*width, value_mean,
                                  width, yerr=numpy.array([value_std,]).T, ecolor=ecolor,**bar_kwargs))
                else:
                    rects.append(
                        pylab.bar(ind[igp]+ibar*width, value, width, **bar_kwargs))
        if legend_props is not None:
            pylab.gca().legend( rects, bar_labels, **legend_props)
    pylab.gca().set_xticks(ind+(nbars-1)*width/2.)
    xlabels = [ group_names[key1] for key1 in keys1 ]
    if not xticklabels_kwargs:
        xticklabels_kwargs = {}
    pylab.gca().set_xticklabels(xlabels,**xticklabels_kwargs)
    pylab.xlim(ind[0]-width,ind[-1]+width*nbars*1.5)
    return rects

