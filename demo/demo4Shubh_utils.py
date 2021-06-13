__author__ = 'jruffio'

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from glob import glob
import os
from copy import copy
import ctypes
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from scipy.interpolate import InterpolatedUnivariateSpline
import multiprocessing as mp
import pandas as pd
import itertools
from scipy.optimize import lsq_linear
from scipy import interpolate
from astropy import constants as const
from scipy.interpolate import interpn
from scipy.interpolate import interp1d
from scipy.special import loggamma

try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

def aper_spec(cube,aper=10,center=None):

    nz,ny,nx = cube.shape
    im = np.nansum(cube,axis=0)
    if center is None:
        kcen, lcen = np.unravel_index(np.nanargmax(im),(ny,nx))
    else:
        lcen,kcen  = center
    x_vec, y_vec = np.arange(nx * 1.)-lcen,np.arange(ny* 1.)-kcen
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)
    r_grid = np.sqrt(x_grid**2+y_grid**2)

    where_aper = np.where(r_grid<aper)
    standard_spec = np.nansum(cube[:,where_aper[0],where_aper[1]],axis=1)

    return standard_spec

def return_64x19(cube):
    #cube should be nz,ny,nx
    if np.size(cube.shape) == 3:
        _,ny,nx = cube.shape
    else:
        ny,nx = cube.shape
    onesmask = np.ones((64,19))
    if (ny != 64 or nx != 19):
        mask = copy(cube).astype(np.float)
        mask[np.where(mask==0)]=np.nan
        mask[np.where(np.isfinite(mask))]=1
        if np.size(cube.shape) == 3:
            im = np.nansum(mask,axis=0)
        else:
            im = mask
        ccmap =np.zeros((3,3))
        for dk in range(3):
            for dl in range(3):
                ccmap[dk,dl] = np.nansum(im[dk:np.min([dk+64,ny]),dl:np.min([dl+19,nx])]*onesmask[0:(np.min([dk+64,ny])-dk),0:(np.min([dl+19,nx])-dl)])
        dk,dl = np.unravel_index(np.nanargmax(ccmap),ccmap.shape)
        if np.size(cube.shape) == 3:
            return cube[:,dk:(dk+64),dl:(dl+19)]
        else:
            return cube[dk:(dk+64),dl:(dl+19)]
    else:
        return cube

def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.nanargmax(cum_posterior)
    if len(x[0:np.min([argmax_post+1,len(x)])]) < 2:
        lx = x[0]
    else:
        tmp_cumpost = cum_posterior[0:np.min([argmax_post+1,len(x)])]
        tmp_x= x[0:np.min([argmax_post+1,len(x)])]
        deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
        try:
            whereinflection = np.where(deriv_tmp_cumpost<0)[0][0]
            where2keep = np.where((tmp_x<=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except:
            pass
        lf = interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[0])
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx=x[-1]
    else:
        tmp_cumpost = cum_posterior[argmax_post::]
        tmp_x= x[argmax_post::]
        deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
        try:
            whereinflection = np.where(deriv_tmp_cumpost>0)[0][0]
            where2keep = np.where((tmp_x>=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except:
            pass
        rf = interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[-1])
        rx = rf(1-0.6827)

    # plt.plot(x,posterior/np.max(posterior))
    # plt.fill_betweenx([0,1],[lx,lx],[rx,rx],alpha=0.5)
    # plt.show()
    return x[argmax_post],x[argmax_post]-lx,rx-x[argmax_post]

def read_sinfoni(filename,cenfilename = None):
    """
    Read sinfoni spectral cube
    """
    with pyfits.open(filename) as hdulist:
        cube = hdulist[0].data
        prihdr = hdulist[0].header
    noisecube = np.ones(cube.shape)
    badpixcube = np.ones(cube.shape)
    badpixcube[np.where(np.isnan(cube))] = np.nan
    badpixcube[0:50,:,:] = np.nan
    badpixcube[440:465,:,:] = np.nan
    badpixcube[2150::,:,:] = np.nan

    nz,ny,nx = cube.shape
    ref_wv = prihdr["CRVAL3"] # wv for slice CRPIX3 in mum
    ref_wv_id = prihdr["CRPIX3"]-1
    dwv = prihdr["CDELT3"] # wv interval between 2 slices in mum
    wvs=np.linspace(ref_wv-dwv*ref_wv_id,ref_wv+dwv*(nz-ref_wv_id),nz,endpoint=False)

    VLT = EarthLocation.from_geodetic(lat=-24.6275 * u.deg, lon=-70.4044 * u.deg, height=2635 * u.m)
    sc = SkyCoord(float(prihdr["RA"]) * u.deg, float(prihdr["DEC"]) * u.deg, frame='icrs')
    barycorr = sc.radial_velocity_correction(obstime=Time(float(prihdr["MJD-OBS"]), format="mjd", scale="utc"),
                                             location=VLT)
    baryrv = barycorr.to(u.km / u.s).value
    # print(prihdr["MJD-OBS"])

    if cenfilename is not None:
        cent_arr = np.genfromtxt(cenfilename, delimiter=',',skip_header=1,usecols=(1,2))
    else:
        cent_arr = None

    return wvs, cube, noisecube, badpixcube, baryrv,cent_arr

def read_osiris(filename,skip_baryrv=False):
    """
    Read OSIRIS spectral cube
    """
    with pyfits.open(filename) as hdulist:
        prihdr = hdulist[0].header
        curr_mjdobs = prihdr["MJD-OBS"]
        cube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        cube = return_64x19(cube)
        noisecube = np.rollaxis(np.rollaxis(hdulist[1].data,2),2,1)
        noisecube = return_64x19(noisecube)
        # cube = np.moveaxis(cube,0,2)
        badpixcube = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
        badpixcube = return_64x19(badpixcube)
        # badpixcube = np.moveaxis(badpixcube,0,2)
        badpixcube = badpixcube.astype(dtype=ctypes.c_double)
        badpixcube[np.where(badpixcube==0)] = np.nan
        badpixcube[np.where(badpixcube!=0)] = 1

    nz,ny,nx = cube.shape
    init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
    dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
    wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

    if not skip_baryrv:
        keck = EarthLocation.from_geodetic(lat=19.8283 * u.deg, lon=-155.4783 * u.deg, height=4160 * u.m)
        sc = SkyCoord(float(prihdr["RA"]) * u.deg, float(prihdr["DEC"]) * u.deg)
        barycorr = sc.radial_velocity_correction(obstime=Time(float(prihdr["MJD-OBS"]), format="mjd", scale="utc"),
                                                 location=keck)
        baryrv = barycorr.to(u.km / u.s).value
    else:
        baryrv = None

    return wvs, cube, noisecube, badpixcube, baryrv

def get_spline_model(x_knots,x_samples,spline_degree=3):
    M = np.zeros((np.size(x_samples),(np.size(x_knots))))
    for chunk in range(np.size(x_knots)):
        tmp_y_vec = np.zeros(np.size(x_knots))
        tmp_y_vec[chunk] = 1
        spl = InterpolatedUnivariateSpline(x_knots, tmp_y_vec, k=spline_degree, ext=0)
        M[:,chunk] = spl(x_samples)
    return M




def _task_convolve_spectrum(paras):
    indices,wvs,spectrum,R = paras

    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    med_dwv = np.median(dwvs)
    for l,k in enumerate(indices):
        pwv = wvs[k]
        FWHM = pwv/R
        sig = FWHM/(2*np.sqrt(2*np.log(2)))
        w = int(np.round(sig/med_dwv*10.))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
        gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
        conv_spectrum[l] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)
        # import matplotlib.py
        # plt.figure(2)
        # plt.plot(stamp_wvs,gausskernel)
        # plt.plot(stamp_wvs,stamp_spec)
        # print(conv_spectrum[l])
        # plt.show()
        # exit()
    return conv_spectrum

def convolve_spectrum(wvs,spectrum,R,mypool=None):
    if mypool is None:
        return _task_convolve_spectrum((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,R))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
        outputs_list = mypool.map(_task_convolve_spectrum, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(R)))
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum


def _arraytonumpy(shared_array, shape=None, dtype=None):
    """
    Covert a shared array to a numpy array
    Args:
        shared_array: a multiprocessing.Array array
        shape: a shape for the numpy array. otherwise, will assume a 1d array
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        numpy_array: numpy array for vectorized operation. still points to the same memory!
                     returns None is shared_array is None
    """
    if dtype is None:
        dtype = ctypes.c_float

    # if you passed in nothing you get nothing
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array.get_obj(), dtype=dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array

from scipy.stats import median_absolute_deviation
def _task_findbadpix(paras):
    data_arr,noise_arr,badpix_arr,med_spec,M_spline = paras
    new_data_arr = np.array(copy(data_arr), '<f4')#.byteswap().newbyteorder()
    new_badpix_arr = copy(badpix_arr)
    res = np.zeros(data_arr.shape) + np.nan
    for k in range(data_arr.shape[1]):
        where_data_finite = np.where(np.isfinite(badpix_arr[:,k])*np.isfinite(data_arr[:,k])*np.isfinite(noise_arr[:,k])*(noise_arr[:,k]!=0))
        if np.size(where_data_finite[0]) == 0:
            res[:,k] = np.nan
            continue
        d = data_arr[where_data_finite[0],k]
        d_err = noise_arr[where_data_finite[0],k]

        M = M_spline[where_data_finite[0],:]*med_spec[where_data_finite[0],None]


        bounds_min = [0, ]* M.shape[1]
        bounds_max = [np.inf, ] * M.shape[1]
        p = lsq_linear(M/d_err[:,None],d/d_err,bounds=(bounds_min, bounds_max)).x
        # p,chi2,rank,s = np.linalg.lstsq(M/d_err[:,None],d/d_err,rcond=None)
        m = np.dot(M,p)
        res[where_data_finite[0],k] = d-m

        # where_bad = np.where((np.abs(res[:,k])>3*np.nanstd(res[:,k])) | np.isnan(res[:,k]))
        where_bad = np.where((np.abs(res[:,k])>3*median_absolute_deviation(res[where_data_finite[0],k])) | np.isnan(res[:,k]))
        new_badpix_arr[where_bad[0],k] = np.nan
        where_bad = np.where(np.isnan(np.correlate(new_badpix_arr[:,k] ,np.ones(2),mode="same")))
        new_badpix_arr[where_bad[0],k] = np.nan
        new_data_arr[where_bad[0],k] = np.nan

        new_data_arr[:,k] = np.array(pd.DataFrame(new_data_arr[:,k]).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:, 0]

    return new_data_arr,new_badpix_arr,res

def findbadpix(cube, noisecube=None, badpixcube=None,chunks=20,mypool=None,med_spec=None):


    if noisecube is None:
        noisecube = np.ones(cube.shape)
    if badpixcube is None:
        badpixcube = np.ones(cube.shape)

    new_cube = copy(cube)
    new_badpixcube = copy(badpixcube)
    nz,ny,nx = cube.shape
    res = np.zeros(cube.shape) + np.nan

    x = np.arange(nz)
    x_knots = x[np.linspace(0,nz-1,chunks+1,endpoint=True).astype(np.int)]
    M_spline = get_spline_model(x_knots,x,spline_degree=3)

    N_valid_pix = ny*nx
    if med_spec is None:
        med_spec = np.nanmedian(cube,axis=(1,2))
    new_badpixcube[np.where(cube==0)] = np.nan

    # plt.plot(med_spec)
    # plt.show()


    if mypool is None:
        data_list = np.reshape(new_cube,(nz,nx*ny))
        noise_list = np.reshape(noisecube,(nz,nx*ny))
        badpix_list = np.reshape(new_badpixcube,(nz,nx*ny))
        out_data,out_badpix,out_res = _task_findbadpix((data_list,noise_list,badpix_list,med_spec,M_spline))
        new_cube = np.reshape(new_cube,(nz,ny,nx))
        new_badpixcube = np.reshape(new_badpixcube,(nz,ny,nx))
        res = np.reshape(out_res,(nz,ny,nx))
    else:
        numthreads = mypool._processes
        chunk_size = N_valid_pix//(3*numthreads)
        wherenotnans = np.where(np.nansum(np.isfinite(badpixcube),axis=0)!=0)
        row_valid_pix = wherenotnans[0]
        col_valid_pix = wherenotnans[1]
        N_chunks = N_valid_pix//chunk_size

        row_indices_list = []
        col_indices_list = []
        data_list = []
        noise_list = []
        badpix_list = []
        for k in range(N_chunks-1):
            _row_valid_pix = row_valid_pix[(k*chunk_size):((k+1)*chunk_size)]
            _col_valid_pix = col_valid_pix[(k*chunk_size):((k+1)*chunk_size)]

            row_indices_list.append(_row_valid_pix)
            col_indices_list.append(_col_valid_pix)

            data_list.append(cube[:,_row_valid_pix,_col_valid_pix])
            noise_list.append(noisecube[:,_row_valid_pix,_col_valid_pix])
            badpix_list.append(new_badpixcube[:,_row_valid_pix,_col_valid_pix])

        _row_valid_pix = row_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix]
        _col_valid_pix = col_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix]

        row_indices_list.append(_row_valid_pix)
        col_indices_list.append(_col_valid_pix)

        data_list.append(cube[:,_row_valid_pix,_col_valid_pix])
        noise_list.append(noisecube[:,_row_valid_pix,_col_valid_pix])
        badpix_list.append(new_badpixcube[:,_row_valid_pix,_col_valid_pix])

        outputs_list = mypool.map(_task_findbadpix, zip(data_list,noise_list,badpix_list,
                                                               itertools.repeat(med_spec),
                                                               itertools.repeat(M_spline)))
        for row_indices,col_indices,out in zip(row_indices_list,col_indices_list,outputs_list):
            out_data,out_badpix,out_res = out
            new_cube[:,row_indices,col_indices] = out_data
            new_badpixcube[:,row_indices,col_indices] = out_badpix
            res[:,row_indices,col_indices] = out_res


    return new_badpixcube, new_cube,res




def _tpool_init_detecplanet(mp_cube,mp_noisecube,mp_badpixcube,mp_res,mp_out,
                              cube_shape,noisecube_shape,badpixcube_shape, res_shape, out_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global gmp_cube,gmp_noisecube,gmp_badpixcube,gmp_res,gmp_out, \
        gcube_shape,gnoisecube_shape,gbadpixcube_shape, gres_shape, gout_shape

    gmp_cube = mp_cube
    gmp_noisecube = mp_noisecube
    gmp_badpixcube = mp_badpixcube
    gmp_res = mp_res
    gmp_out = mp_out

    gcube_shape = cube_shape
    gnoisecube_shape = noisecube_shape
    gbadpixcube_shape = badpixcube_shape
    gres_shape = res_shape
    gout_shape = out_shape

def extract_cube_stamp(cube,center,plx,ply,w):
    nx,ny,nx = cube.shape
    k,l = int(np.round(center[1]+ply)),int(np.round(center[0]+plx))
    return cube[:,
               np.max([k-w,0]):np.min([k+w+1,ny]),
               np.max([l-w,0]):np.min([l+w+1,nx])]

def get_speckle_model(xvec,yvec,zvec,xref,yref,zref,center,Nl,l,Ns,w):
    nz,ny,nx = np.size(zvec),np.size(yvec),np.size(xvec)
    # rref = np.sqrt((xref-center[0])**2+(yref-center[1])**2)
    # thref = np.arctan2(yref,xref)
    # u = np.array([np.cos(thref),np.sin(thref)])
    # v = np.array([-np.sin(thref),np.cos(thref)])

    x_knots = zvec[np.linspace(0,nz-1,Ns,endpoint=True).astype(np.int)]
    M_spline = get_spline_model(x_knots,zvec,spline_degree=2)
    X, Y, Z = np.meshgrid(xvec,yvec,zvec, indexing='ij')

    xp = np.linspace(-Nl*l,Nl*l,2*Nl+1,endpoint=True)
    yp = np.linspace(-Nl*l,Nl*l,2*Nl+1,endpoint=True)
    zp = zvec
    points = (xp, yp, zp)
    # X, Y, Z = np.meshgrid(y, z, x)
    # Xp, Yp, Zp = np.meshgrid(xp,yp,zp, indexing='ij')
    M = np.zeros((nz,(2*w+1),(2*w+1),(2*Nl+1),(2*Nl+1),Ns))
    for dxpref_id,dxpref in enumerate(xp):
        for dypref_id,dypref in enumerate(yp):
            for knot in range(Ns):
                # if dxpref_id != Nl:
                #     continue
                # if dypref_id != Nl:
                #     continue
                # if knot != 2:#Ns-1:
                #     continue
                values = np.zeros((np.size(xp),np.size(yp),np.size(zp)))
                values[dxpref_id,dypref_id,:] = M_spline[:,knot]
                # print(values.shape)
                # plt.figure(3)
                # plt.plot( M_spline[:,knot])
                # # plt.show()
                # print((X[xref,yref,:]-center[0])/Z[xref,yref,:]*zref-(xref-center[0]))
                # print((Y[xref,yref,:]-center[1])/Z[xref,yref,:]*zref-(yref-center[1]))
                # print(Z[xref,yref,:])
                _xp = (X[xref-w:xref+w+1,yref-w:yref+w+1,:]-center[0])/Z[xref-w:xref+w+1,yref-w:yref+w+1,:]*zref-(xref-center[0])
                _yp = (Y[xref-w:xref+w+1,yref-w:yref+w+1,:]-center[1])/Z[xref-w:xref+w+1,yref-w:yref+w+1,:]*zref-(yref-center[1])
                _zp = Z[xref-w:xref+w+1,yref-w:yref+w+1,:]
                point = np.concatenate([np.ravel(_xp)[:,None],np.ravel(_yp)[:,None],np.ravel(_zp)[:,None]],axis=1)
                # point = np.array([-Nl*w,-Nl*w,x_knots[0]])
                f = interpn(points, values, point,bounds_error=False,fill_value=0)
                # print(point[0:10,:])
                # print(f[0:10])
                # print(point[point.shape[0]-10:point.shape[0],:])
                # print(f[point.shape[0]-10:point.shape[0]])
                # print(f.shape)
                f = np.reshape(f,(2*w+1,2*w+1,nz))
                # print(f.shape)
                # plt.figure(1)
                # plt.subplot(1,4,1)
                # plt.imshow(np.moveaxis(f[:,:,0],1,0),origin="lower")
                # plt.clim([-1,1])
                # plt.subplot(1,4,2)
                # plt.imshow(np.moveaxis(f[:,:,nz//2],1,0),origin="lower")
                # plt.clim([-1,1])
                # plt.subplot(1,4,3)
                # plt.imshow(np.moveaxis(f[:,:,1000],1,0),origin="lower")
                # plt.clim([-1,1])
                # plt.subplot(1,4,4)
                # plt.imshow(np.moveaxis(f[:,:,-1],1,0),origin="lower")
                # plt.clim([-1,1])
                # plt.figure(2)
                # print(center,(xref,yref,zref))
                # plt.plot(f[1,1,:])
                # plt.show()
                f = np.moveaxis(f,[0,1,2],[2,1,0])
                M[:,:,:,dypref_id,dxpref_id,knot] = f
                # print(f.shape)
    return M


def pixgauss2d(p,shape,hdfactor=10,xhdgrid=None, yhdgrid=None):
    A,xA,yA,w,bkg = p
    ny,nx = shape
    if xhdgrid is None or yhdgrid is None:
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor*nx).astype(np.float)/hdfactor,np.arange(hdfactor*ny).astype(np.float)/hdfactor)
    else:
        hdfactor = xhdgrid.shape[0]//ny
    gaussA_hd = A/(2*np.pi*w**2)*np.exp(-0.5*((xA-xhdgrid+0.5)**2+(yA-yhdgrid+0.5)**2)/w**2)
    gaussA = np.nanmean(np.reshape(gaussA_hd,(ny,hdfactor,nx,hdfactor)),axis=(1,3))
    return gaussA + bkg

def like_fit_pixgauss2d(p,slice,hdfactor=5):
    # p = A,xA,yA,w,bkg
    res = slice - pixgauss2d(p, slice.shape,hdfactor=hdfactor)
    return np.nansum(res**2)

def LPFvsHPF(myvec,cutoff):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    window = int(round(np.size(myvec_cp)/(cutoff/2.)/2.))#cutoff
    tmp = np.array(pd.DataFrame(np.concatenate([myvec_cp, myvec_cp[::-1]], axis=0)).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))
    myvec_cp_lpf = np.array(pd.DataFrame(tmp).rolling(window=window, center=True).median().interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[0:np.size(myvec), 0]
    myvec_cp[wherenans] = myvec_cp_lpf[wherenans]


    fftmyvec = np.fft.fft(np.concatenate([myvec_cp, myvec_cp[::-1]], axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan

    return LPF_myvec,HPF_myvec


def make_linear_model(cube,noisecube,badpixcube,plx,ply,plrv,center,wvs,transmission,star_spectrum,planet_spec_func,science_baryrv,w=0,psfwidth0=0.75):
    nz,ny,nx = cube.shape

    planet_spec = transmission*planet_spec_func(wvs*(1-(plrv-science_baryrv)/const.c.to('km/s').value))

    nospec_planet_model = np.zeros((nz,2*w+1,2*w+1))
    hdfactor = 5
    xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor*(2*w+1)).astype(np.float)/hdfactor,np.arange(hdfactor*(2*w+1)).astype(np.float)/hdfactor)
    nospec_planet_model += pixgauss2d([1.,w,w,psfwidth0,0.],(2*w+1,2*w+1),xhdgrid=xhdgrid, yhdgrid=yhdgrid)[None,:,:]
    nospec_planet_model = nospec_planet_model/np.nansum(nospec_planet_model,axis=(1,2))[:,None,None]

    planet_spec_LPF,planet_spec_HPF = LPFvsHPF(planet_spec,40)
    planet_model_LPF = nospec_planet_model*planet_spec_LPF[:,None,None]
    planet_model_HPF = nospec_planet_model*planet_spec_HPF[:,None,None]

    k,l = int(np.round(center[1]+ply)),int(np.round(center[0]+plx))
    oldM = np.zeros(nospec_planet_model.shape)
    for m in range((2*w+1)):
        for n in range((2*w+1)):
            oldM[:,m,n] = LPFvsHPF((star_spectrum[0]*LPFvsHPF(cube[:,k-w+m,l-w+n],40)[0]/LPFvsHPF(star_spectrum[0],40)[0]),40)[1]

    HPFmodel = np.concatenate([planet_model_HPF[:,:,:, None],oldM[:,:,:, None]], axis=3)
    
    return HPFmodel


def _task_detecplanet(plx_chunk,ply_chunk,plxid_chunk,plyid_chunk,plrvvec,center,dtype,
                          wvs,telluric_transmission,star_spectrum,
                    planet_spec_func,science_bary_rv,w,psfwidth0):
    global gmp_cube,gmp_noisecube,gmp_badpixcube,gmp_res,gmp_out, \
        gcube_shape,gnoisecube_shape,gbadpixcube_shape, gres_shape, gout_shape
    cube_np = _arraytonumpy(gmp_cube, gcube_shape,dtype=dtype)
    noisecube_np = _arraytonumpy(gmp_noisecube, gnoisecube_shape,dtype=dtype)
    badpixcube_np = _arraytonumpy(gmp_badpixcube, gbadpixcube_shape,dtype=dtype)
    res_np = _arraytonumpy(gmp_res, gres_shape,dtype=dtype)
    out_np = _arraytonumpy(gmp_out, gout_shape,dtype=dtype)
    nz,ny,nx = cube_np.shape

    for rvid,rv in enumerate(plrvvec):
        for plx,ply,plxid,plyid in zip(plx_chunk,ply_chunk,plxid_chunk,plyid_chunk):
            print(rv, plx, ply, rvid, plxid, plyid)
            # out_np[:,rvid,plyid,plxid] = 1
            # continue
            k,l = int(np.round(center[1]+ply)),int(np.round(center[0]+plx))

            M = make_linear_model(cube_np,noisecube_np,badpixcube_np,plx,ply,rv,center,
                          wvs,telluric_transmission,star_spectrum,planet_spec_func,science_bary_rv,w,psfwidth0)
            nz_stamp,ny_stamp,nx_stamp,N_linpara = M.shape

            cube_stamp_hpf = copy(cube_np[:,k-w:k+w+1,l-w:l+w+1])
            for yid in range(cube_stamp_hpf.shape[1]):
                for xid in range(cube_stamp_hpf.shape[2]):
                    cube_stamp_hpf[:,yid,xid] = LPFvsHPF(cube_stamp_hpf[:,yid,xid],40)[1]
            stamp_cube_ravel = np.ravel(cube_stamp_hpf)

            stamp_noisecube_ravel = np.ravel(noisecube_np[:,k-w:k+w+1,l-w:l+w+1])
            stamp_badpixcube_ravel = np.ravel(badpixcube_np[:,k-w:k+w+1,l-w:l+w+1])
            M_ravel = np.reshape(M,(nz_stamp*ny_stamp*nx_stamp,N_linpara))

            where_finite = np.where(np.isfinite(stamp_badpixcube_ravel))
            # print(np.size(where_finite[0]))
            if np.size(where_finite[0]) <= 0.5*(nz_stamp*ny_stamp*nx_stamp):
                continue
            data=stamp_cube_ravel[where_finite]
            sigmas=stamp_noisecube_ravel[where_finite]
            M_ravel = M_ravel[where_finite[0],:]
            N_data = M_ravel.shape[0]
            logdet_Sigma = np.sum(2 * np.log(stamp_noisecube_ravel))

            bounds_min = [-np.inf, ] +[0, ]* (M_ravel.shape[1]-1)
            bounds_max = [np.inf, ] * M_ravel.shape[1]

            # plt.plot(data,label="data")
            # plt.show()

            M_ravel_norm = M_ravel/ sigmas[:, None]
            try:
            # if 1:
                paras = lsq_linear(M_ravel_norm, data / sigmas,bounds=(bounds_min, bounds_max)).x
                # print(paras[0])

                model = np.dot(M_ravel , paras)
                residuals = data  - model
                chi2 = np.nansum((residuals/sigmas) ** 2)

                canvas = np.zeros((nz,2*w+1,2*w+1))
                canvas.shape = (nz*(2*w+1)*(2*w+1),)
                canvas[where_finite] = residuals
                canvas.shape = (nz,2*w+1,2*w+1)
                res_np[:,k,l] = np.nanmean(canvas,axis=(1,2))

                covphi = chi2 / N_data * np.linalg.inv(np.dot(M_ravel_norm.T, M_ravel_norm))
                slogdet_icovphi0 = np.linalg.slogdet(np.dot(M_ravel_norm.T, M_ravel_norm))

                paras_H0 = lsq_linear(M_ravel_norm[:,1::], data / sigmas,bounds=(bounds_min[1::], bounds_max[1::])).x
                model_H0 = np.dot(M_ravel[:,1::] , paras_H0)
                residuals_H0 = data  - model_H0
                chi2_H0 = np.nansum((residuals_H0/sigmas) ** 2)
                slogdet_icovphi0_H0 = np.linalg.slogdet(np.dot(M_ravel_norm[:,1::].T, M_ravel_norm[:,1::]))

                out_np[0,rvid,plyid,plxid] = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0[1] - (N_data-1+N_linpara-1)/2*np.log(chi2)+loggamma((N_data-1+N_linpara-1)/2)+(N_linpara-N_data)/2*np.log(2*np.pi)
                out_np[1,rvid,plyid,plxid] = -0.5*logdet_Sigma - 0.5*slogdet_icovphi0_H0[1] - (N_data-1+N_linpara-1-1)/2*np.log(chi2_H0)+loggamma((N_data-1+(N_linpara-1)-1)/2)+((N_linpara-1)-N_data)/2*np.log(2*np.pi)
                out_np[2:(N_linpara+2),rvid,plyid,plxid] = paras
                out_np[(N_linpara+2):(2*N_linpara+2),rvid,plyid,plxid] = np.sqrt(np.diag(covphi))



                if 0:# PLot debug
                    print(paras)
                    plt.figure(1)
                    plt.plot(data,label="data")
                    plt.plot(model,label="model (planet + speckle)")
                    plt.plot(model_H0,label="model H0")
                    plt.plot(paras[0]*M_ravel[:,0],label="planet model")
                    plt.plot(residuals,label="Residuals")
                    plt.legend()

                    plt.figure(2)
                    residuals[np.where(np.isnan(residuals))] = 0
                    # _res = np.ones(_res.shape)
                    res_ccf = np.correlate(residuals,residuals,mode="same")/np.size(residuals)
                    res_ccf_argmax = np.nanargmax(res_ccf)
                    plt.plot(np.arange(-100,101),res_ccf[(res_ccf_argmax-100):(res_ccf_argmax+101)]/res_ccf[res_ccf_argmax])
                    plt.ylim([-0.05,0.5])

                    plt.show()
            except:
                out_np[:,rvid,plyid,plxid] = np.nan
    return

def detecplanet(cube,w,center=None,plxvec=None,plyvec=None,plrvvec=None, noisecube=None, badpixcube=None,numthreads=16,dtype= ctypes.c_double,
                      wvs=None,telluric_transmission=None,star_spectrum=None,
                    planet_spec_func=None,science_bary_rv=None,psfwidth0=None):

    nz,ny,nx = cube.shape
    if plxvec is None:
        plxvec = np.arange(nx)
    if plyvec is None:
        plyvec = np.arange(ny)
    if plrvvec is None:
        plrvvec = np.array([0])


    cube_pad = np.pad(cube,((0,0),(w,w),(w,w)),mode="constant",constant_values=0)
    nz,ny,nx = cube_pad.shape

    if noisecube is None:
        noisecube_pad = np.ones(cube_pad.shape)
    else:
        noisecube_pad = np.pad(noisecube,((0,0),(w,w),(w,w)),mode="constant",constant_values=np.inf)
    if badpixcube is None:
        badpixcube_pad = np.ones(cube_pad.shape)
    else:
        badpixcube_pad = np.pad(badpixcube,((0,0),(w,w),(w,w)),mode="constant",constant_values=np.nan)
    if center is None:
        center_pad = [w,w]
    else:
        center_pad = [center[0]+w,center[1]+w]

    M = make_linear_model(cube_pad,noisecube_pad,badpixcube_pad,plxvec[0],plyvec[0],plrvvec[0],center_pad,
                          wvs,telluric_transmission,star_spectrum,planet_spec_func,science_bary_rv,w,psfwidth0)
    nz_stamp,ny_stamp,nx_stamp,N_linpara = M.shape

    res = np.zeros(cube_pad.shape) + np.nan


    plx_grid, ply_grid = np.meshgrid(plxvec, plyvec)
    plxid_grid,plyid_grid = np.meshgrid(np.arange(np.size(plxvec)), np.arange(np.size(plyvec)))
    N_loc = np.size(plx_grid)
    plx_grid_ravel,ply_grid_ravel = np.ravel(plx_grid),np.ravel(ply_grid)
    plxid_grid_ravel,plyid_grid_ravel = np.ravel(plxid_grid),np.ravel(plyid_grid)
    if numthreads != 0:
        chunk_size =np.max([N_loc//(3*numthreads),1])

        N_chunks = N_loc//chunk_size
        plx_chunk_list = []
        ply_chunk_list = []
        plxid_chunk_list = []
        plyid_chunk_list = []
        for k in range(N_chunks-1):
            plx_chunk_list.append(plx_grid_ravel[(k*chunk_size):((k+1)*chunk_size)])
            ply_chunk_list.append(ply_grid_ravel[(k*chunk_size):((k+1)*chunk_size)])
            plxid_chunk_list.append(plxid_grid_ravel[(k*chunk_size):((k+1)*chunk_size)])
            plyid_chunk_list.append(plyid_grid_ravel[(k*chunk_size):((k+1)*chunk_size)])
        plx_chunk_list.append(plx_grid_ravel[((N_chunks-1)*chunk_size):N_loc])
        ply_chunk_list.append(ply_grid_ravel[((N_chunks-1)*chunk_size):N_loc])
        plxid_chunk_list.append(plxid_grid_ravel[((N_chunks-1)*chunk_size):N_loc])
        plyid_chunk_list.append(plyid_grid_ravel[((N_chunks-1)*chunk_size):N_loc])


    mp_cube = mp.Array(dtype, np.size(cube_pad))
    cube_shape = cube_pad.shape
    cube_np = _arraytonumpy(mp_cube, cube_shape,dtype=dtype)
    cube_np[:] = copy(cube_pad)
    mp_noisecube = mp.Array(dtype, np.size(noisecube_pad))
    noisecube_shape = noisecube_pad.shape
    noisecube_np = _arraytonumpy(mp_noisecube, noisecube_shape,dtype=dtype)
    noisecube_np[:] = noisecube_pad
    mp_badpixcube = mp.Array(dtype, np.size(badpixcube_pad))
    badpixcube_shape = badpixcube_pad.shape
    badpixcube_np = _arraytonumpy(mp_badpixcube, badpixcube_shape,dtype=dtype)
    badpixcube_np[:] = copy(badpixcube_pad)
    mp_res = mp.Array(dtype, np.size(res))
    res_shape = res.shape
    res_np = _arraytonumpy(mp_res, res_shape,dtype=dtype)
    res_np[:] = np.nan

    mp_out = mp.Array(dtype, ((1+1+2*N_linpara)*np.size(plrvvec)*np.size(plx_grid)))
    out_shape = [(1+1+2*N_linpara),np.size(plrvvec),plx_grid.shape[0],plx_grid.shape[1]]
    # dim 1: likelihood + best lin para values + uncertainties for lin para values
    # dim 2: RVs
    # dim 3: locations
    out_np = _arraytonumpy(mp_out, out_shape,dtype=dtype)
    out_np[:] = np.nan


    if numthreads == 0:
        _tpool_init_detecplanet(mp_cube,mp_noisecube,mp_badpixcube,mp_res,mp_out,
                                  cube_shape,noisecube_shape,badpixcube_shape, res_shape,out_shape)
        _task_detecplanet(plx_grid_ravel,ply_grid_ravel,plxid_grid_ravel,plyid_grid_ravel,plrvvec,center_pad,dtype,
                          wvs,telluric_transmission,star_spectrum,planet_spec_func,science_bary_rv,w,psfwidth0)
    else:
        # _tpool_init_detecplanet(mp_cube,mp_noisecube,mp_badpixcube,mp_res,mp_out,
        #                           cube_shape,noisecube_shape,badpixcube_shape, res_shape,out_shape)
        # for plx_chunk,ply_chunk,plxid_chunk,plyid_chunk  in zip(plx_chunk_list, ply_chunk_list,plxid_chunk_list,plyid_chunk_list):
        #     _task_detecplanet(plx_chunk,ply_chunk,plxid_chunk,plyid_chunk,plrvvec,center_pad,dtype,
        #                   wvs,telluric_transmission,star_spectrum,planet_spec_func,science_bary_rv,w,psfwidth0,chunks)
        # print(len(plx_chunk_list))
        # exit()
        tpool = mp.Pool(processes=numthreads, initializer=_tpool_init_detecplanet,
                        initargs=(mp_cube,mp_noisecube,mp_badpixcube,mp_res,mp_out,
                                  cube_shape,noisecube_shape,badpixcube_shape, res_shape,out_shape),
                        maxtasksperchild=50)
        tasks = [tpool.apply_async(_task_detecplanet, args=(plx_chunk,ply_chunk,plxid_chunk,plyid_chunk,plrvvec,center_pad,dtype,
                                                            wvs,telluric_transmission,star_spectrum,planet_spec_func,science_bary_rv,w,psfwidth0))
                 for plx_chunk,ply_chunk,plxid_chunk,plyid_chunk  in zip(plx_chunk_list, ply_chunk_list,plxid_chunk_list,plyid_chunk_list)]

        #save it to shared memory
        for taskid, task in enumerate(tasks):
            # print("Finished image chunk {0}/{1}".format(taskid,len(row_indices_list)))
            task.wait()
        tpool.close()
        tpool.join()

    return out_np,res_np



