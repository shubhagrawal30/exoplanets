__author__ = 'jruffio'

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from glob import glob
import os
from astropy import constants as const
import multiprocessing as mp
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
from astropy import units as u
import scipy.io as scio
from copy import copy
from scipy.optimize import minimize
import warnings
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

import demo4Shubh_utils as utils


#------------------------------------------------
if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass
    warnings.filterwarnings('ignore')

    main_dir = "./"


    R=4000
    numthreads=16

    science_dir = os.path.join(main_dir,"kap_And/20161106/reduced_jb/")
    filelist = glob(os.path.join(science_dir,"*_020.fits"))
    tel_stand_dir = os.path.join(main_dir,"kap_And/20161106/reduced_telluric_jb/HIP_111538/")
    tel_filelist = glob(os.path.join(tel_stand_dir,"*_020.fits"))

    filelist.sort()
    tel_filelist.sort()

    sc_filename = filelist[1]
    # pl_center = [10,37]
    # for sc_filename in filelist:
    if 1:
        mypool = mp.Pool(processes=numthreads)

        ##################
        ## Process telluric standards
        standard_spec_list = []
        for filename in tel_filelist:
            wvs, standard_cube, standard_noisecube, standard_badpixcube, standard_bary_rv = utils.read_osiris(filename,skip_baryrv = True)
            standard_spec_list.append(utils.aper_spec(standard_cube,aper=2,center=None))
        med_standard_spec = np.nanmedian(standard_spec_list,axis=0)

        standard_spec_list = []
        gw_list = []
        for filename in tel_filelist:
            wvs, standard_cube, standard_noisecube, standard_badpixcube, standard_bary_rv = utils.read_osiris(filename)

            standard_badpixcube,corr_standard_cube,standard_res = utils.findbadpix(standard_cube, noisecube=standard_noisecube, badpixcube=standard_badpixcube,chunks=20,mypool=mypool,med_spec=med_standard_spec)
            standard_spec = utils.aper_spec(corr_standard_cube,aper=5,center=None)
            standard_spec_list.append(standard_spec)

            standard_im = np.nanmean(corr_standard_cube,axis=0)
            ycen,xcen = np.unravel_index(np.nanargmax(standard_im),standard_im.shape)
            w_guess = 1
            para0 = [1,xcen,ycen,w_guess,0]
            bounds = [(0,np.inf),(xcen-2,xcen+2),(ycen-2,ycen+2),(0.1*w_guess,w_guess*5),(-np.inf,np.inf)]
            A,xA,yA,gw,bkg = minimize(utils.like_fit_pixgauss2d,para0,bounds=bounds,args=(standard_im,5),options={"maxiter":1e5}).x
            gw_list.append(gw)

        # Width of the spatial PSF
        gw = np.median(gw_list)
        # Combined spectrum of the telluric standard star
        mn_standard_spec = np.nanmean(np.array(standard_spec_list),axis=0)

        ## Process telluric standards
        phoenix_folder = "./planets_templates"
        phoenix_A0_filename = glob(os.path.join(phoenix_folder, "kap_And_lte11600-4.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"))[0]
        phoenix_wv_filename = os.path.join(phoenix_folder, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        # host_rv = -12.7 #+-0.8
        # host_limbdark = 0.5
        # host_vsini = 150 #unknown
        standard_rv = -3#+-4.4
        standard_limbdark = 0.5
        standard_vsini = 60

        with pyfits.open(phoenix_wv_filename) as hdulist:
            phoenix_wvs = hdulist[0].data / 1.e4
        crop_phoenix = np.where((phoenix_wvs > wvs[0] - (wvs[-1] - wvs[0]) / 4) * (phoenix_wvs < wvs[-1] + (wvs[-1] - wvs[0]) / 4))
        phoenix_wvs = phoenix_wvs[crop_phoenix]
        with pyfits.open(phoenix_A0_filename) as hdulist:
            phoenix_A0 = hdulist[0].data[crop_phoenix]
        phoenix_A0_func = interp1d(phoenix_wvs,phoenix_A0,bounds_error=False,fill_value=np.nan)
        wvs4broadening = np.arange(phoenix_wvs[0],phoenix_wvs[-1],1e-4)
        # Rotational broadening: broadening of the stellar lines due to the star's rotation (ie spin).
        broadened_phoenix_A0 = pyasl.rotBroad(wvs4broadening, phoenix_A0_func(wvs4broadening), standard_limbdark, standard_vsini)
        broadened_phoenix_A0 = utils.convolve_spectrum(wvs4broadening,broadened_phoenix_A0,R,mypool=mypool)
        phoenix_A0_func = interp1d(wvs4broadening/(1-(standard_rv+standard_bary_rv)/const.c.to('km/s').value),broadened_phoenix_A0,bounds_error=False,fill_value=np.nan)

        # telluric spectrum (ie absorption of the Earth atmosphere)
        telluric_transmission =  mn_standard_spec/phoenix_A0_func(wvs)
        # plt.plot(wvs,telluric_transmission)
        # plt.show()
        ##################

        ##################
        ## Read Science cube
        print(sc_filename)
        wvs, cube, noisecube, badpixcube, science_bary_rv = utils.read_osiris(sc_filename) # 36,12
        badpixcube,corr_cube,res_cube = utils.findbadpix(cube, noisecube=noisecube, badpixcube=badpixcube,chunks=20,mypool=mypool,med_spec=mn_standard_spec)
        ##################

        ##################
        ## Build a model spectrum of the star in the science cube
        nz,ny,nx=cube.shape
        chunks = 20
        x = np.arange(nz)
        x_knots = x[np.linspace(0,nz-1,chunks+1,endpoint=True).astype(np.int)]
        M_spline = utils.get_spline_model(x_knots,x,spline_degree=3)
        from scipy.optimize import lsq_linear
        fit_cube = np.zeros(corr_cube.shape) + np.nan
        fit_cube_noise = np.zeros(corr_cube.shape) + np.nan
        for k in range(ny):
            for l in range(nx):
                print(k,l)
                where_data_finite = np.where(np.isfinite(badpixcube[:,k,l])*np.isfinite(corr_cube[:,k,l])*np.isfinite(noisecube[:,k,l])*(noisecube[:,k,l]!=0))
                d = mn_standard_spec[where_data_finite]
                d_err = noisecube[where_data_finite[0],k,l]

                M = M_spline[where_data_finite[0],:]*corr_cube[where_data_finite[0],k,l][:,None]
                bounds_min = [0, ]* M.shape[1]
                bounds_max = [np.inf, ] * M.shape[1]
                p = lsq_linear(M/d_err[:,None],d/d_err,bounds=(bounds_min, bounds_max)).x
                m = np.dot(M,p)

                fit_cube[where_data_finite[0],k,l] = m
                fit_cube_noise[where_data_finite[0],k,l] = d_err*np.sqrt(np.nanmean(((d-m)/d_err)**2))
        star_spectrum = np.nansum(fit_cube/fit_cube_noise**2,axis=(1,2))/np.nansum(1/fit_cube_noise**2,axis=(1,2))
        ##################

        ##################
        ## Build a model spectrum for the planet (atmospheric models)
        travis_spec_filename=os.path.join(main_dir,"planets_templates",
                                          "KapAnd_lte19-3.50-0.0.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019.7.save")
        travis_spectrum = scio.readsav(travis_spec_filename)
        ori_planet_spec = np.array(travis_spectrum["f"])
        wmod = np.array(travis_spectrum["w"])/1.e4
        crop_model = np.where((wmod > wvs[0] - (wvs[-1] - wvs[0]) / 4) * (wmod < wvs[-1] + (wvs[-1] - wvs[0]) / 4))
        wmod = wmod[crop_model]
        ori_planet_spec = ori_planet_spec[crop_model]
        broadened_planet_spec = ori_planet_spec#pyasl.rotBroad(wmod,ori_planet_spec, 0.5, 40)
        # mypool = mp.Pool(processes=numthreads)
        broadened_planet_spec = utils.convolve_spectrum(wmod,broadened_planet_spec,R,mypool=mypool)
        mypool.close()
        mypool.join()

        # planet model spectrum
        planet_spec_func = interp1d(wmod,broadened_planet_spec,bounds_error=False,fill_value=np.nan)

        # plt.plot(planet_spec_func(wvs))
        # plt.show()
        ##################

        w = 0 # width of the data stamp
        center= [0,0]
        # plxvec,plyvec = np.array([10]),np.array([37])
        plxvec,plyvec = np.arange(0,19,1),np.arange(0,64,1)
        plrvvec = np.array([-10])
        # plrvvec = np.linspace(-1000,1000,21,endpoint=True)

        out,res = utils.detecplanet(corr_cube,w, center=center,plxvec=plxvec,plyvec=plyvec,plrvvec=plrvvec,
                                           noisecube=noisecube, badpixcube=badpixcube,numthreads=0,
                          wvs=wvs,telluric_transmission=telluric_transmission,star_spectrum=[star_spectrum],
                        planet_spec_func=planet_spec_func,science_bary_rv=science_bary_rv,psfwidth0=gw)
        print(out.shape)
        print(res.shape)
        nz,ny,nx = res.shape

        N_linpara = (out.shape[0]-2)//2
        rvid0 = np.argmin(np.abs(plrvvec))

        plt.figure(1)
        snr_map = out[2,rvid0,:,:]/out[2+N_linpara,rvid0,:,:]
        bayes_factor_ratio_map = out[0,rvid0,:,:]-out[1,rvid0,:,:]#np.exp(out[0,rvid0,:,:]-out[1,rvid0,:,:])
        snr_scaling = np.nanstd(snr_map[0:30,5:15])
        print(snr_scaling)
        plt.imshow(snr_map/snr_scaling,origin="lower",interpolation="nearest")
        plt.clim([-5,10])
        plt.colorbar()

        plt.show()
        exit()
