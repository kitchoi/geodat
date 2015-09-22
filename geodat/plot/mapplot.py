import numpy as np
import copy
from mpl_toolkits.basemap import Basemap, shiftgrid

def MapSetup(f):
    def plot(lons,lats,field,basemap_kwargs=None,*args,**kwds):
        '''
        Input:
            field       - either a numpy array or a util.nc.Variable
                          if field is a numpy.array, arguments "lat" and "lon" will be needed.
            lats         - numpy.array for the latitude (assumed to be in degree)
            lons         - numpy.array for the longitude (assumed to be in degree, and is monotonic)
            lon_0       - center of the map, default = 180.
            projection  - projection option for Basemap, default = 'cyl'
            draw_lon    - draw meridians (default=True)
            draw_lat    - draw parallels (default=True)
            parallels   - arrays for the parallels (if undefined, it will be automatically determined)
            meridians   - arrays for the meridians (if undefined, it will be automatically determined)
            basemap_kwargs (dict): parsed to Basemap while setting up the map
        Examples:
            this_function(lons,lats,numpy.array,...)
        '''
        lons = copy.deepcopy(lons)
        projection = kwds.get('projection','cyl')
        lon_0 = kwds.get('lon_0',180.)
        londiff = np.diff(lons)
        if not all(londiff>0.):  # not montonically increasing
            idiff = np.where(londiff<0.)[0][0]
            step = londiff[idiff]*-1.
            lons[idiff:] += step
        
        llcrnrlon = kwds.get('llcrnrlon',lons[0])
        urcrnrlon = kwds.get('urcrnrlon',lons[-1])
        llcrnrlat = kwds.get('llcrnrlat',lats[0])
        urcrnrlat = kwds.get('urcrnrlat',lats[-1])
        
        if basemap_kwargs is None:
            basemap_kwargs = {}
        m = Basemap(projection=projection,lon_0=lon_0,
                    llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,
                    urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,**basemap_kwargs)
        x,y = m(*np.meshgrid(lons,lats))
        plotf = getattr(m,f.__name__)
        # closest 10
        closest10 = lambda a : a - np.mod(a,10.)
        parallels = kwds.pop('parallels',sorted(list(set(closest10(np.arange(m.latmin-5.,m.latmax+5.,(m.latmax-m.latmin)/5.))))))
        meridians = kwds.pop('meridians',sorted(list(set(closest10(np.arange(m.lonmin-5.,m.lonmax+5.,(m.lonmax-m.lonmin)/5.))))))
        drawlat = kwds.pop('draw_lat',True)
        drawlon = kwds.pop('draw_lon',True)
        cs = plotf(x,y,field,*args,**kwds)
        if drawlat: m.drawparallels(parallels,labels=[1,0,0,0],linewidth=0)
        if drawlon: m.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0)
        return m,cs
    return plot

contour = MapSetup(Basemap.contour)
