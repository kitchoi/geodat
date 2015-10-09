import logging

import numpy

import geodat.arrays

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    import spharm
    _SPHARM_INSTALLED = True
    _import_error = None
except ImportError:
    logger.warning("Failed to load spharm")
    _SPHARM_INSTALLED = False
    _import_error = ImportError("Failed to load spharm")

radius =  6371200. # in m
beta = 2*numpy.pi*2./86400./radius

def grid_degree(NY=256,NX=None):
    '''
    Return lon,lat for the spherical grid with (NX,NY)
    If NX is not given, default NX = NY*2
    Spherical harmonic examples:
    NY = 256
    NY = 128
    '''
    if NX is None: NX = NY*2
    if not _SPHARM_INSTALLED:
        raise _import_error
    latdeg,wgt = spharm.gaussian_lats_wts(NY)
    londeg = numpy.linspace(0.,360.,NX+1)[:-1]
    return londeg,latdeg[::-1]


def grid_xy(NY=256,M=170):
    '''
    Return x,y (in meter) for the spherical grid with (NX=NY*2,NY)
    Spherical harmonic examples:
    NY = 256, M = 170
    NY = 128, M = 85
    '''
    londeg,latdeg = grid_degree(NY,M)

    lon = numpy.pi * londeg / 180.
    lat = numpy.pi * latdeg / 180.
    x = radius*numpy.cos(lat[:,numpy.newaxis])*lon[numpy.newaxis,:]
    y = radius*lat[:,numpy.newaxis]*numpy.ones((len(lat),len(lon)))
    return x,y


def regrid(lon_in,lat_in,data,nlon_o,nlat_o):
    ''' Given data(nlat,nlon), regrid the data into (nlat_o,nlon_o)
    using the spharm.regrid routine
    If the input data is not on a complete sphere, copy the input
    onto a complete sphere before regridding

    Input:
    lon_in - the longitude the input data is on
    lat_in - the latitude the input data is on
    data   - numpy.ndarray
    nlon_o - integer
    nlat_o - integer

    Return:
    numpy.ndarray of shape (nlat_o,nlon_o)

    '''
    if not _SPHARM_INSTALLED:
        raise _import_error

    assert data.ndim <= 3

    # determine if the domain is a complete sphere
    dlon = numpy.diff(lon_in[0:2])
    AllLon = dlon*(len(lon_in)+1) > 360.
    dlat = numpy.diff(lat_in[0:2])
    AllLat = dlat*(len(lat_in)+1) > 180.
    sliceobj = [slice(None),]*data.ndim
    if AllLon and AllLat:  # The data covers the whole sphere
        inputdata = data
        nlat_i = len(lat_in)
        nlon_i = len(lon_in)
        lat = lat_in
        lon = lon_in
    else:
        inputdata,lon,lat = regional2global(data,lon_in,lat_in)
        nlat_i = len(lat)
        nlon_i = len(lon)

    gridin = spharm.Spharmt(nlon_i,nlat_i)
    gridout = spharm.Spharmt(nlon_o,nlat_o)

    return spharm.regrid(gridin,gridout,inputdata)


def regional2global(data,lon_in,lat_in):
    # expand the domain for the input data
    dlat = numpy.diff(lat_in).mean()
    dlon = numpy.diff(lon_in).mean()
    nlat = int(180./dlat)
    nlon = int(360./dlon)
    lon,lat = grid_degree(nlat,nlon)
    latshift = lat[numpy.abs(lat - lat_in.min()).argmin()] -\
               lat_in.min()
    lonshift = lon[numpy.abs(lon - lon_in.min()).argmin()] -\
               lon_in.min()
    lon = lon - lonshift
    lat = lat - latshift
    nlon_i = len(lon)
    nlat_i = len(lat)
    selector = geodat.arrays.getSlice
    originalshape = list(data.shape)
    originalshape[0] = nlat_i
    originalshape[1] = nlon_i
    globaldata = numpy.zeros(originalshape)
    sliceobj = [slice(None),]*data.ndim
    sliceobj[0] = selector(lat,lat_in.min()-dlat/2.,lat_in.max()+dlat/2.)
    sliceobj[1] = selector(lon,lon_in.min()-dlon/2.,lon_in.max()+dlon/2.,modulo=360.)
    sliceobj = tuple(sliceobj)
    globaldata[sliceobj] = data
    return globaldata,lon,lat
