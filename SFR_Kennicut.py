#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:15:00 2023

@author: tauro
"""
#import python libraries
import numpy as np 
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os 
#from astropy.nddata import Cutout2D
from scipy import ndimage
from scipy import interpolate
from scipy import stats
import astropy.constants as K
import astropy.units as u
from astropy.cosmology import Planck15 as p15
import scipy.ndimage
import scipy.interpolate
from math import degrees, radians
from scipy.signal import savgol_filter
from lmfit import minimize, Parameters, report_fit
import scipy.special as special
from regriding import regrid2d
import astropy.wcs as wcs
import Polygon


# from scipy.optimize import curve_fit
# from reproject import reproject_interp
#----------------------------------------------------------------------------

def regrid2d(xcor_in, ycor_in, data_in, variance_in, quality_in, weights_in, xcor_out, ycor_out,
                divide_by_pixel_area=True, dorner_variance_scaling=True,
                print_progress=False, logger=None, logging_level='info'):


    """ Method to project data from a continuous input grid of 2D pixels on a continuous output 2D grid of pixels.
​
    This function loops through each pixel of the input and output grids, determines the fraction of the area of
    the input pixel covered by the output pixel (if any) and uses this value to assign a fraction of the value of
    the input pixel to the output pixel. It uses the Polygon package to compute the area of the overlap between a
    pair of input and output pixels.
    The code is derived from nips.module.rect_module code written by Nora Luetzgendorf and follows the equations
    (4.2) from the PhD thesis of Bernard Dorner(2012).
​
    Parameters
    ----------
    xcor_in : numpy.ndarray
        2D float array containing the x-axis coordinates of the corners of the input pixel grid (must be continuous).
        Shape: (nx_in+1, ny_in+1) where (nx_in, ny_in) is the shape of the input data array.
    ycor_in : numpy.ndarray
        2D float array containing the y-axis coordinates of the corners of the input pixel grid (must be continuous).
        Shape: (nx_in+1, ny_in+1) where (nx_in, ny_in) is the shape of the input data array.
    data_in: numpy.ndarray or numpy.ma.array
        2D float array containing the input data. Data must be per unit of area.
        Shape: (nx_in, ny_in).
    variance_in: numpy.ndarray or numpy.ma.array
        2D array containing the input data variance information. Data must be per (unit of area)^2.
        Shape: (nx_in, ny_in).
    quality_in: numpy.ndarray or numpy.ma.array
        2D array containing the input data quality flags. Must be of type numpy.uint64.
        Shape: (nx_in, ny_in).
    weights_in: numpy.ndarray or numpy.ma.array
        2D array containing user-defined weights that will be applied to the input data and variance.
        Shape: (nx_in, ny_in).
    xcor_out : numpy.ndarray
        2D float array containing the x-axis coordinates of the corners of the output pixel grid (must be continuous).
        Shape: (nx_out+1, ny_out+1) where (nx_out, ny_out) is the shape of the ouput data array.
    ycor_out : numpy.ndarray
        2D float array containing the y-axis coordinates of the corners of the output pixel grid (must be continuous).
        Shape: (nx_out+1, ny_out+1) where (nx_out, ny_out) is the shape of the output data array.
    divide_by_pixel_area : bool
        if set to True, the data will be divided by the input pixel area and the variance by the square of the
        pixel area. Default: True.
    dorner_variance_scaling : bool
        if set to True, the fraction of the input pixel area will be used instead of its square when projecting the
        variance following equation 4.2 of Bernhard Dorner's PhD. Default: True.
    print_progress : bool
        if set to True, an ASCII progress bar will be printed on the standard output. Default: False.
​
    Returns
    -------
    A tuple (data_out, variance_out, quality_out, filling_factor_out, npix_out) of 5 numpy arrays of size
    (nx_out,ny_out).
    data_out : output data array. Note that if divide_by_pixel_area=True, the output data will be per units of
        pixel area.
    variance_out : output variance array. Note that the weighting scheme applied when projecting the variance
        depends on whether dorner_variance_scaling is True or False.
    quality_out : output quality array. It is the bitwise OR of the quality flags of all the input pixels
        contributing to the output pixel. The function does not perform any quality flagging for the projection
        itself (e.g. based on the the filling factor of the output pixel).
    filling_factor_out : output array containing the fraction of the output pixel area covered by the overlapping
        input pixels.
    npix_out : output array indicating how many (valid = unmasked) input pixels overlapped with the output pixel.
​
    """
    # =============================================================================================================
    # Paranoid checks
    # =============================================================================================================

    shape_in = np.shape(data_in)
    nx_in = shape_in[0]
    ny_in = shape_in[1]
    ma_data_in = np.ma.array(data_in).harden_mask()
    # -------------------------------------------------------------------------------------------------------------
    ma_variance_in = np.ma.array(variance_in).harden_mask()
    # ------------------------------------------------------------------------------------------------------------
    ma_quality_in = np.ma.array(quality_in).harden_mask()
    # -------------------------------------------------------------------------------------------------------------
    ma_weights_in = np.ma.array(weights_in).harden_mask()
    # -------------------------------------------------------------------------------------------------------------
    work = np.shape(xcor_out)
    if (xcor_out[-1,0] >= xcor_out[0,0]):
        xdir_out = 1
    else:
        xdir_out = -1
    nx_out = work[0] - 1
    ny_out = work[1] - 1
    shape_out = (nx_out, ny_out)
    # -------------------------------------------------------------------------------------------------------------
    if (ycor_out[0,-1] >= ycor_out[0,0]):
        ydir_out = 1
    else:
        ydir_out = -1

    # =============================================================================================================
    # Extracting some information from the xcor_out and ycor_out arrays
    # =============================================================================================================
    xmax_out = xcor_out.max()
    ymax_out = ycor_out.max()
    xmin_out = xcor_out.min()
    ymin_out = ycor_out.min()
    envelope_out = Polygon.Polygon(((xmin_out, ymin_out), (xmax_out, ymin_out),
                                    (xmax_out, ymax_out), (xmin_out, ymax_out)))
    # =============================================================================================================
    # Main loops
    # =============================================================================================================
    combined_mask_in = np.ma.getmaskarray(ma_data_in) | np.ma.getmaskarray(ma_variance_in) \
                    | np.ma.getmaskarray(ma_quality_in) | np.ma.getmaskarray(ma_weights_in)
    corners_pixel_in = np.zeros((4, 2))
    corners_pixel_out = np.zeros((4, 2))
    data_out = np.zeros(shape_out, dtype=ma_data_in.dtype)
    variance_out = np.zeros(shape_out, dtype=ma_variance_in.dtype)
    quality_out = np.zeros(shape_out, dtype=ma_quality_in.dtype)
    normfactor_out = np.zeros(shape_out, dtype=ma_data_in.dtype)
    normfactorsquare_out = np.zeros(shape_out, dtype=ma_data_in.dtype)
    filling_factor_out = np.zeros(shape_out, dtype=ma_data_in.dtype)
    npix_out = np.zeros(shape_out, dtype=int)
    ncount = 0
    # -------------------------------------------------------------------------------------------------------------
    # Loop on the input pixels - y-axis
    # -------------------------------------------------------------------------------------------------------------
    for iy_in in range(ny_in):
        if ((combined_mask_in[:, iy_in] == True).all()):
            continue
        # ---------------------------------------------------------------------------------------------------------
        # Loop on the input pixels - x-axis
        # ---------------------------------------------------------------------------------------------------------
        for ix_in in range(nx_in):
            # .....................................................................................................
            # Checking if the pixel is masked
            # .....................................................................................................
            if combined_mask_in[ix_in, iy_in]:
                continue
            # .....................................................................................................
            # Input weight
            # .....................................................................................................
            w_in = ma_weights_in[ix_in, iy_in]
            if w_in == 0:
                continue
            # .....................................................................................................
            # Initialising the polygon corresponding to the input pixel
            # .....................................................................................................
            corners_pixel_in[0, 0] = xcor_in[ix_in, iy_in]
            corners_pixel_in[1, 0] = xcor_in[ix_in + 1, iy_in]
            corners_pixel_in[2, 0] = xcor_in[ix_in + 1, iy_in + 1]
            corners_pixel_in[3, 0] = xcor_in[ix_in, iy_in + 1]
            corners_pixel_in[0, 1] = ycor_in[ix_in, iy_in]
            corners_pixel_in[1, 1] = ycor_in[ix_in + 1, iy_in]
            corners_pixel_in[2, 1] = ycor_in[ix_in + 1, iy_in + 1]
            corners_pixel_in[3, 1] = ycor_in[ix_in, iy_in + 1]
            polygon_pixel_in = Polygon.Polygon(corners_pixel_in)
            # Skip immediately if the input pixel is outside of the envelope of the output grid
            if ((envelope_out & polygon_pixel_in).area() == 0.0):
                continue
            area_pixel_in = polygon_pixel_in.area()
            # .....................................................................................................
            # Storing the input pixel values in work variables
            # .....................................................................................................
            if divide_by_pixel_area:
                d_in = data_in[ix_in, iy_in] / area_pixel_in
                v_in = variance_in[ix_in, iy_in] / area_pixel_in**2
            else:
                d_in = data_in[ix_in, iy_in]
                v_in = variance_in[ix_in, iy_in]
            q_in = quality_in[ix_in, iy_in]
            # .....................................................................................................
            # Isolating the output pixels potentially overlapping with the input one
            # .....................................................................................................
            bb_in = polygon_pixel_in.boundingBox()
            if (xdir_out == 1):
                if (ydir_out == 1):
                    valid_pixels_out = np.where((xcor_out[0:-1, 0:-1] <= bb_in[1]) &
                                                   (xcor_out[1:, 1:] >= bb_in[0]) &
                                                   (ycor_out[0:-1, 0:-1] <= bb_in[3]) &
                                                   (ycor_out[1:, 1:] >= bb_in[2]))
                else:
                    valid_pixels_out = np.where((xcor_out[0:-1, 0:-1] <= bb_in[1]) &
                                                   (xcor_out[1:, 1:] >= bb_in[0]) &
                                                   (ycor_out[0:-1, 0:-1] >= bb_in[2]) &
                                                   (ycor_out[1:, 1:] <= bb_in[3]))
            else:
                if (ydir_out == 1):
                    valid_pixels_out = np.where((xcor_out[0:-1, 0:-1] >= bb_in[0]) &
                                                   (xcor_out[1:, 1:] <= bb_in[1]) &
                                                   (ycor_out[0:-1, 0:-1] <= bb_in[3]) &
                                                   (ycor_out[1:, 1:] >= bb_in[2]))
                else:
                    valid_pixels_out = np.where((xcor_out[0:-1, 0:-1] >= bb_in[0]) &
                                                   (xcor_out[1:, 1:] <= bb_in[1]) &
                                                   (ycor_out[0:-1, 0:-1] >= bb_in[2]) &
                                                   (ycor_out[1:, 1:] <= bb_in[3]))

            # -----------------------------------------------------------------------------------------------------
            # Loop over the output pixels
            # -----------------------------------------------------------------------------------------------------
            for ix_out,iy_out in zip(valid_pixels_out[0], valid_pixels_out[1]):
                # .................................................................................................
                # Generating the polygon corresponding to the output pixel
                # .................................................................................................
                corners_pixel_out[0, 0] = xcor_out[ix_out, iy_out]
                corners_pixel_out[1, 0] = xcor_out[ix_out + 1, iy_out]
                corners_pixel_out[2, 0] = xcor_out[ix_out + 1, iy_out + 1]
                corners_pixel_out[3, 0] = xcor_out[ix_out, iy_out + 1]
                corners_pixel_out[0, 1] = ycor_out[ix_out, iy_out]
                corners_pixel_out[1, 1] = ycor_out[ix_out + 1, iy_out]
                corners_pixel_out[2, 1] = ycor_out[ix_out + 1, iy_out + 1]
                corners_pixel_out[3, 1] = ycor_out[ix_out, iy_out + 1]
                polygon_pixel_out = Polygon.Polygon(corners_pixel_out)
                area_pixel_out = polygon_pixel_out.area()
                # .................................................................................................
                # Projection
                # .................................................................................................
                oa = (polygon_pixel_in & polygon_pixel_out).area()
                if oa == 0.0:
                    continue
                filling_factor_out[ix_out, iy_out] = filling_factor_out[ix_out, iy_out] + oa / area_pixel_out
                npix_out[ix_out, iy_out] = npix_out[ix_out, iy_out] + 1
                fa = oa / area_pixel_in
                current_normfactor_out = normfactor_out[ix_out, iy_out] + fa * w_in
                data_out[ix_out, iy_out] = (d_in * fa * w_in + data_out[ix_out, iy_out]
                                            * normfactor_out[ix_out, iy_out]) / current_normfactor_out
                if (dorner_variance_scaling):
                    variance_out[ix_out, iy_out] = (v_in * fa * (w_in)**2 + variance_out[ix_out, iy_out]
                                            * normfactor_out[ix_out, iy_out]**2) / current_normfactor_out**2
   
                else:
                    variance_out[ix_out, iy_out] = (v_in * (fa * w_in)**2 + variance_out[ix_out, iy_out]
                                          * normfactor_out[ix_out, iy_out]**2) / current_normfactor_out**2
   
                quality_out[ix_out, iy_out] = quality_out[ix_out, iy_out] | q_in
                normfactor_out[ix_out, iy_out] = current_normfactor_out
    # =============================================================================================================
    # This is the end, my friend, the end
    # =============================================================================================================
    return (data_out, variance_out, quality_out, filling_factor_out, npix_out)
#---------------------------------------------------------------------- -----  

path = os.path.dirname(os.path.abspath('__file__'))

# Open and managing the datacubes
filefits_data = 'NGC6810_crop.fits'
filefits_antenna = 'NGC6810_antenna.fits'
datacube = fits.open(path+'/file/'+filefits_data)[0]
datacube_antenna = fits.open(path+'/file/'+filefits_antenna)[0]
datacube.data = np.squeeze(datacube.data)
datacube_antenna.data = np.squeeze(datacube_antenna.data)
Nz,Ny,Nx = datacube.shape
print (Nz, Ny, Nx)


# define the z-axis which corresponds to frequency
naxis3 = datacube.header['NAXIS3']
crpix3 = datacube.header['CRPIX3']
crval3 = datacube.header['CRVAL3']
cdelt3 = datacube.header['CDELT3']

kk = 1+np.arange(naxis3)
            
frequency = crval3+cdelt3*(kk-crpix3) #Hz
frequency /= 1e9 #GHz

print(frequency[:10])


# define the z-axis in velocity units 
# average frequency
frequency_mean = np.mean(frequency)*u.GHz
frequency_mean_err = scipy.stats.sem(frequency) 
print(frequency_mean)


# z = v/c = (nu_emit - nu_obs)/nu_obs 
velocity_unit = ((frequency_mean- (frequency*u.GHz))/(frequency*u.GHz))*K.c.to('km/s')
print(velocity_unit[:10])
velocity = velocity_unit.value
print(velocity[:10])
dv = velocity[0]-velocity[1]

#----------------------------------------------------------------------------

flux = fits.open(path + '/moments_map/NGC6810/flux_crop_clean.fits')[0]  #già moltiplicato per dv
vel = fits.open(path + '/moments_map/NGC6810/vel_crop_clean.fits')[0]
disp = fits.open(path + '/moments_map/NGC6810/vdisp_crop_clean.fits')[0]
velerr = fits.open(path + '/moments_map/NGC6810/vel_err_crop_clean.fits')[0]
disperr = fits.open(path + '/moments_map/NGC6810/vdisp_err_crop_clean.fits')[0]
fluxerr = fits.open(path + '/moments_map/NGC6810/flux_err_crop_clean.fits')[0]

plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(flux.data, origin = 'lower', cmap = 'jet')
plt.colorbar()
plt.subplot(132)
plt.imshow(vel.data, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar()
plt.subplot(133)
plt.imshow(disp.data, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar()
flux = flux.data
vel = vel.data
disp = disp.data
velerr = velerr.data
disperr = disperr.data
fluxerr = fluxerr.data

maskflux  = ~np.isnan(flux)
maskvel = ~np.isnan(vel)
maskdisp = ~np.isnan(disp) 
#----------------------------------------------------------------------------

# Stuff with pixel, beam_area, convertions
cdelt1 = abs(datacube.header['CDELT1'])
print(cdelt1)

cdelt2 = abs(datacube.header['CDELT2'])
print(cdelt1)

cdelt1_deg = cdelt1*u.deg
cdelt1_arcsec = cdelt1_deg.to('arcsec')
print(cdelt1_deg,cdelt1_arcsec)

redshift = 0.006541  #From Simbad Catalogue 
arcsec_to_kpc =  p15.arcsec_per_kpc_proper(redshift)
print(arcsec_to_kpc)

pixel_size_kpc = cdelt1_arcsec/arcsec_to_kpc
print(pixel_size_kpc)

bmaj = abs(datacube.header['BMAJ'])
bmin = abs(datacube.header['BMIN'])

beam_area = np.pi * bmaj * bmin / cdelt1 / cdelt2 / (4*np.log(2))

beam_area_arcsec = beam_area / 3600 *u.arcsec**2

pixel_area_arcsec = cdelt1_arcsec**2
pixel_area_kpc = pixel_size_kpc **2

#----------------------------------------------------------------------------

# Q = maj_semi_axe / min_semi_axe of the ellipse found with Kinemetry
Q = 0.5
ellipticity = 1 - Q
inc = np.arctan(ellipticity)    #Inclination of the Galaxy
vel_rot = vel/np.sin(inc)       #From observated velocity to rotation velocity
#----------------------------------------------------------------------------
# Velocity/Dispersion along ellipses
mean_vel = np.ndarray(110) 
mean_disp = np.ndarray(110)
m_v_err = np.ndarray(110)
m_d_err = np.ndarray(110)
v_d_err = np.ndarray(110)
ell_rad = np.ndarray(110)
ell_rad_err = np.ndarray(110)
mean_flux = np.ndarray(110)
mean_flux_err = np.ndarray(110)

# Define the center of the matrix and the distance from the center
center = (225, 120)
dist_min = 6
dist_max = 8

# Create a meshgrid of x and y coordinates
x, y = np.meshgrid(np.arange(vel.shape[1]), np.arange(vel.shape[0]))

# Calculate the distance of each point from the center along the x and y axis
dist_y = (y - center[0])**2
dist_x = (x - center[1])**2 / (1 - ellipticity)**2
contatore = 0

for k in range(1, 111, 1):
    # Create a mask for points that lie within 2 ellipse of distance = 3 from eachother
    mask1 = dist_x + dist_y <= dist_max**2
    mask2 = dist_x + dist_y > dist_min**2
    mask = np.logical_and(mask1,mask2)
    ell_rad[k-1] = (dist_max + dist_min) / 2
    ell_rad_err[k-1] = (dist_max - dist_min)/2
    
    mean_flux_tmp = np.nanmean((flux[mask]))
    mean_flux[k-1] = mean_flux_tmp                     #Jy *dv / beam
    sem_flux = scipy.stats.sem(flux[mask], nan_policy='omit')
    mean_flux_err[k-1] = sem_flux
    # Calculate the mean velocity along the ellipse
    mean_velocity = np.nanmean(np.abs(vel_rot[mask]))   #capire se con questo metodo devo comunque correggere per l'inclinazione
    mean_dispersion = np.nanmean(np.abs(disp[mask]))
    mean_vel[k-1]=(mean_velocity)
    mean_disp[k-1]=(mean_dispersion)
    sem_vel = scipy.stats.sem(vel[mask], nan_policy='omit')
    sem_disp = scipy.stats.sem(disp[mask], nan_policy='omit')
    m_v_err[k-1] = sem_vel
    m_d_err[k-1] = sem_disp    

    dist_min += 2
    dist_max += 2
    
v_d = mean_vel/mean_disp    
v_d_err_rel = np.sqrt((m_v_err/mean_vel)**2+(m_d_err/mean_disp)**2)
v_d_err = v_d_err_rel * v_d 
ell_rad_kpc = ell_rad*pixel_size_kpc
ell_rad_err_kpc = ell_rad_err * pixel_size_kpc
#----------------------------------------------------------------------------
# PLOTTING V/S, V_mean, S_mean
plt.figure(figsize = (12,4))
# plt.scatter(ell_rad_kpc, mean_vel/mean_disp, label= 'v vs sigma', marker='.')
plt.errorbar(ell_rad_kpc, v_d, v_d_err, fmt='.', label='v vs sigma (Ellipses Version)')
plt.xlabel('R [kpc]')
plt.ylabel('v/s')
plt.title('v/s in the different ellipses')
plt.legend()
plt.show()

plt.figure(figsize = (12,4))
plt.errorbar(ell_rad_kpc, mean_vel, m_v_err, fmt = '.', label = 'Mean velocity in the different ellipses ')
plt.xlabel('R[kpc]')
plt.ylabel('V_m')
plt.title('mean velocity')
plt.legend()
plt.show()

plt.figure(figsize = (12,4))
plt.errorbar(ell_rad_kpc, mean_disp, m_d_err, fmt = '.', label = 'Mean velocity dispersion in the different ellipses')
plt.xlabel('R[kpc]')
plt.ylabel('S_m')
plt.title('mean velocity dispersion')
plt.legend()
plt.show()
#----------------------------------------------------------------------------

# Define the multicomponent model for the rotation curve
def residual(pars, x, data= None):
    y = x/ 2 * pars['disk_scale']
    R0 = pars['R0']
    v_disk2 = 2 * y**2 * K.G.to('kpc3 kg-1 s-2').value * pars['M_disk'] * (special.i0(y) *special.k0(y) - special.i1(y) * special.k1(y)) / (pars['disk_scale'] * (1 - np.exp(-R0/pars['disk_scale']) *(1 + R0/pars['disk_scale'])))
    v_disk2 = 1 * v_disk2 * u.kpc**2 / u.s**2
    v_disk2 = 1 * v_disk2.to('km2 s-2').value
    n = 1
    b = 2*n - 1/3 + 0.009876/n
    p = 1 - 0.6097/n + 0.05563/n**2
    x_b = b *(x/pars['r_e'])**(1/n)                                                                                                                   
    v_bulge2 = K.G.to('kpc3 kg-1 s-2').value *pars['M_bulge'] / x * special.gammainc(n*(3-p), x_b) / special.gamma(n*(3-p))
    v_bulge2 = 1 * v_bulge2 * u.kpc**2 / u.s**2
    v_bulge2 = 1 * v_bulge2.to('km2 s-2').value
    H = 69.6 * u.km /u.s / u.Mpc
    H = 1 * H.to('s-1').value
    rho_c = 3 * H**2 / (8*np.pi * K.G.to('kpc3 kg-1 s-2').value)
    M_200 = 200 * rho_c *4 * np.pi * pars['r_200']**3 / 3   #kg
    # M_200 = pars['M_200']
    x_r = x / pars['r_200']
    V_200 = np.sqrt(K.G.to('kpc3 kg-1 s-2').value * M_200 / pars['r_200'])  #kpc s-1
    v_dm2 = V_200 **2 * (1/x_r)*(np.log(1 + x_r * pars['c']) -(pars['c']*x_r)/(1 + pars['c']*x_r)) / (np.log(1 + pars['c']) - pars['c']/(1+ pars['c']))
    v_dm2 = 1* v_dm2 *u.kpc**2/u.s**2
    v_dm2 = 1* v_dm2.to('km2 s-2').value
    model = np.sqrt(np.abs(v_disk2) + np.abs(v_bulge2) + np.abs(v_dm2))
    if data is None:
            return model, np.sqrt(v_disk2), np.sqrt(v_bulge2), np.sqrt(v_dm2)
    return model - data


# Load the data
xdata, ydata = ell_rad_kpc.value, mean_vel

fit_params = Parameters()
fit_params.add('R0', value = 3, min= 0.01, max = 20)
fit_params.add('disk_scale', value=2.8, min = 2.7, max= 2.9)
fit_params.add('M_disk', value=3e+10 * K.M_sun.value, min = 2.2e+10 * K.M_sun.value, max = 6e10 * K.M_sun.value)
fit_params.add('r_e', value=0.46, min = 0.45, max = 0.47)
fit_params.add('M_bulge', value=4e+10 * K.M_sun.value, min = 2.7e+10 * K.M_sun.value, max = 6e10 * K.M_sun.value)
fit_params.add('r_200', value=350, min = 340, max = 360)
fit_params.add('c', value=4, min = 2, max = 30000)
# fit_params.add('M_200', value = 3e+37, min = 0.01)

out = minimize(residual, fit_params, args=(xdata,), nan_policy='omit' , kws={'data': ydata})
fit, v_disk, v_bulge, v_dm = residual(out.params, xdata)
print('##')
print('Rotation Curve Fit')
report_fit(out)

plt.figure(figsize = (12,4))
plt.errorbar(xdata, ydata, m_v_err, fmt = '.', label = 'Data ')
plt.plot(xdata, fit, label='best fit')
plt.plot(xdata, v_disk, label='Disk component')
plt.plot(xdata, v_bulge, label='Bulge component')
plt.plot(xdata, v_dm, label='Dark Matter component')
plt.xlabel('radius[Kpc]')
plt.ylabel('velocity[km/s]')
plt.title('Velocity curve rotation fit')


plt.legend()
plt.show()

#----------------------------------------------------------------------------


# Luminosity Distance
D = 28.3 * u.Mpc #Mpc with H_o = 69.6 and flat universe

# Mean frequency
freq = frequency_mean.value #Ghz

# CO Luminosity_map from flux
masknan = ~np.isnan(flux)
L_CO = 3.25e7 * flux * D**2 / freq**2 / (1+redshift)**3
L_CO_err = L_CO.value * np.sqrt((fluxerr/flux)**2 + (frequency_mean_err/frequency_mean.value)**2)

# CO to H2 Convertion Factor
alpha1 = 0.8 * K.M_sun #/(K km s^-1 pc^2)
alpha2 = 1.2 * K.M_sun 
alpha3= 4.4 * K.M_sun

# Gass Mass
M_out1 = alpha1.value * L_CO.value #M_sun
M_out2 = alpha2.value * L_CO.value
M_out3 = alpha3.value * L_CO.value
M_out1_err = alpha1 * L_CO_err #M_sun
M_out2_err = alpha2 * L_CO_err
M_out3_err = alpha3 * L_CO_err

#Using log to avoid overflow error on the surface density
logm1 = np.log10(M_out1)
logm2 = np.log10(M_out2)
logm3 = np.log10(M_out3)
logpixel = np.log10((beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec).value)

# Surface Density
E1 = np.full_like(flux, np.nan,dtype=np.float128)
E2 = np.full_like(flux,np.nan, dtype=np.float128)
E3 = np.full_like(flux,np.nan, dtype=np.float128)
E1[masknan] =  M_out1[masknan] / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)  # kg/kpc^2
E2[masknan] = M_out2[masknan] / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)
E3[masknan] =  M_out3[masknan] / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)
logE1 = logm1 - logpixel
logE2 = logm2 - logpixel
logE3 = logm3 - logpixel
test = np.nansum(np.exp(logE1))
test = test / (u.kpc)**2
test = 1*test.to('pc^-2')
test1 = np.log(test.value)
test = np.nansum(np.exp(logE2))
test = test / (u.kpc)**2
test = 1*test.to('pc^-2')
test2 = np.log(test.value)
test = np.nansum(np.exp(logE3))
test = test / (u.kpc)**2
test = 1*test.to('pc^-2')
test3 = np.log(test.value)
# E_gas_tot = np.nansum(np.exp(logE1))
# logE_gas_tot = np.log10(E_gas_tot)
logE_gas_tot_08 = 1* test1 # pc^-2
logE_gas_tot_12 = 1* test2 # pc^-2
logE_gas_tot_44 = 1* test3 # pc^-2

del(ell_rad, ell_rad_kpc)



# Open and managing the datacubes from MUSE
filefits_data = 'NGC6810_Data.fits'
datacube_muse = fits.open(path+'/file/Muse/'+filefits_data)[0]
Nz,Ny,Nx = datacube_muse.shape
# print (Nz, Ny, Nx)


# define the z-axis which corresponds to frequency
naxis3 = datacube_muse.header['NAXIS3']
crpix3 = datacube_muse.header['CRPIX3']  # reference pixel
crval3 = datacube_muse.header['CRVAL3']  # angstrom (Starting wavelenght)
cdelt3 = datacube_muse.header['CD3_3']   # wavelenght increment

kk = 1+np.arange(naxis3)
            
wavelenght = crval3+cdelt3*(kk-crpix3) #A
wavelenght_m = (wavelenght / 1e10 ) * u.m

wavelenght_beta = wavelenght_m[8:98]
wavelenght_O = wavelenght_m[110:210]
wavelenght_alpha = wavelenght_m[1400:1500]

frequency = K.c.to('m s-1') / wavelenght_m #Hz
frequency = 1 * frequency.to('THz')

frequency_beta = frequency[8:98]
frequency_O = frequency[110:210]
frequency_alpha = frequency[1400:1500]


# define the z-axis in velocity units 
# average frequency
frequency_mean = np.mean(frequency)
# print(frequency_mean)


# z = v/c = (nu_emit - nu_obs)/nu_obs 
velocity_unit = ((frequency_mean- (frequency))/(frequency))*K.c.to('km/s')
velocity_beta = (((612.604172*u.THz - (frequency_beta))/(frequency_beta))*K.c.to('km/s')).value
velocity_O = (((594.850769*u.THz - (frequency_O))/(frequency_O))*K.c.to('km/s')).value
velocity_alpha = ((( 453.778836*u.THz - (frequency_alpha))/(frequency_alpha))*K.c.to('km/s')).value
# print(velocity_unit[:10])
velocity = velocity_unit.value
# print(velocity[:10])
dv = np.abs(velocity[1]-velocity[0])
dlambda = np.abs(wavelenght[1]-wavelenght[0])

#----------------------------------------------------------------------------

# TOTAL SPECTRUM
# location of the target
x0,y0 = 155, 150
# size of the square aperture 
dl = 80
# extract the spectrum
spectrum = np.nansum(datacube_muse.data[:,y0-dl:y0+dl,x0-dl:x0+dl],axis = (1,2))

# # 0plot: Wavelenght - Spectrum
# plt.figure(figsize = (12,4))
# plt.plot(wavelenght, spectrum, label = 'data')
# plt.plot(wavelenght,wavelenght*0,':',color = 'black')
# plt.xlabel('wavelenght [A°]')
# plt.ylabel('flux ')
# plt.title('Total Spectrum')
# plt.legend()
# plt.show()

# # 1plot: frequency - spectrum
# plt.figure(figsize = (12,4))
# plt.plot(frequency, spectrum, label = 'data')
# plt.plot(frequency,frequency*0,':',color = 'black')
# plt.xlabel('frequency [THz]')
# plt.ylabel('flux ')
# plt.title('Total Spectrum')
# plt.legend()
# plt.show()



#----------------------------------------------------------------------------

## RMS DETERMINATION WITH THE POWER RESPONSE 

# Choosing an empty region
x0, y0 = 280, 30
dl = 15
noise = datacube_muse.data[:,y0-dl:y0+dl,x0-dl:x0+dl]
error_beta = np.std(noise[8:98,:,:])
error_O = np.std(noise[ 110:210,:,:])
error_alpha = np.std(noise[1400:1500, :, :])
error_tot = np.std(noise[:, :, :])


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# Making 3 different subcubes for the 3 part of the fit (H_beta, O_III doublet, H_alpha e NII doublet)

#H_beta

O_III = datacube_muse.data[8:98, :, :]
w_beta = wavelenght[8:98]
O_III = datacube_muse.data[110:210, :, :]
w_O = wavelenght[110:210]
H_alpha = datacube_muse.data[1400:1500, :, :]
w_alpha = wavelenght[1400:1500]


#----------------------------------------------------------------------------


flux_H_alpha = fits.open(path + '/flux_map_H_definitive_dust_corrected.fits')[0].data
amperr =fits.open(path + '/6amperr_H.fits')[0].data
flux_H_alpha *= 1e-20 
# Plot            
plt.figure(figsize = (14,9))
plt.imshow(flux_H_alpha, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'Flux [Jy]')
plt.title('H_alpha')


#----------------------------------------------------------------------------

## L_H_alpha and Star Formation Rate 

# Stuff with pixel, beam_area, convertions
cdelt1 = abs(datacube_muse.header['CD1_1'])
print(cdelt1)

cdelt2 = abs(datacube_muse.header['CD2_2'])
print(cdelt1)

cdelt1_deg = cdelt1*u.deg
cdelt1_arcsec = cdelt1_deg.to('arcsec')
print(cdelt1_deg,cdelt1_arcsec)

redshift = 0.006541  #From Simbad Catalogue 
arcsec_to_kpc =  p15.arcsec_per_kpc_proper(redshift)
print(arcsec_to_kpc)

pixel_size_kpc = cdelt1_arcsec/arcsec_to_kpc
print(pixel_size_kpc)


pixel_area_arcsec = cdelt1_arcsec**2
pixel_area_kpc = pixel_size_kpc **2

# Luminosity Distance
D = 28.3 * u.Mpc #Mpc with H_o = 69.6 and flat universe
d_l = D.to('cm') # Distance in cm 

ell_rad = np.ndarray(74)

L_H_map = 4 * np.pi * d_l**2 * flux_H_alpha   #erg/s
sfr_map = L_H_map / (1.26 * 10**41) # (M_sun yr^-1)
sfr_map = sfr_map * K.M_sun # kg yr^-1
# sfr_map = L_H_map / (1.26 * 10**41) 
log_sfr = np.log10(sfr_map.value)
log_pixel = np.log10(pixel_area_kpc.value)
# E_sfr_map = sfr_map / pixel_area_kpc # kg yr^-1 kpc^-2
logE_sfr_map = log_sfr - log_pixel

L_H = 4 * np.pi * d_l**2 * np.nansum(flux_H_alpha)   #total flux
sfr = L_H / (1.26 * 10**41) # (M_sun yr^-1)
sfr = sfr * K.M_sun # kg yr^-1
E_sfr = (sfr / pixel_area_kpc).value # kg yr^-1 kpc^-2
logE_sfr = np.log10(E_sfr)

# Plot            
plt.figure(figsize = (14,9))
plt.imshow(logE_sfr_map, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'log SFR Density [Kg yr^-1 kpc^-2')
plt.title('Star Formation Rate Density')
plt.show()

plt.figure(figsize = (8,14))
plt.imshow(logE1, origin = 'lower', cmap = 'jet', vmin = 36, vmax = 41)
plt.colorbar(shrink = 0.7, label = 'log Gas Surface density [Kg kpc^-2')
plt.title('Gas Surface Density')
plt.show()



#----------------------------------------------------------------------------


from astropy.coordinates import SkyCoord

# Set up the header information
ra_center = 19.4333 * u.deg
dec_center = -58.65577777778 * u.deg
pix_scale = 0.2 * u.arcsec

# Load the x and y pixel coordinates from the alma_datacube
# x_pixels = np.linspace(0,449,450) 
# y_pixels = np.linspace(0,239,240) 
x_pixels = np.linspace(0,319,320) 
y_pixels = np.linspace(0,317,318)  

# Convert the pixel coordinates to celestial coordinates
ra, dec = [], []
for x, y in zip(x_pixels, y_pixels):
    # Calculate the relative offset from the center of the datacube
    dx = (x - len(x_pixels)/2) * pix_scale.to(u.deg)
    dy = (y - len(y_pixels)/2) * pix_scale.to(u.deg)

    # Calculate the absolute celestial coordinates
    coord = SkyCoord(ra_center + dx, dec_center + dy, frame='icrs')
    ra.append(coord.ra.deg)
    dec.append(coord.dec.deg)

# ra and dec are now arrays of celestial coordinates in degrees



##Coordinates in arcsec
del datacube.header['*3*']
del datacube.header['*4*']
datacube.header['NAXIS'] = 2

# alma_wcs = wcs.WCS(datacube.header, naxis = [2,1])
alma_wcs = wcs.WCS(datacube.header, naxis = [2,1])
muse_wcs = wcs.WCS(datacube_muse.header, naxis= [2,1])

plt.figure(figsize = (16,8))
ax1 = plt.subplot(121,projection=alma_wcs,slices=('y', 'x'))
im1 = ax1.imshow(logE1,origin = 'lower')
dec = ax1.coords[1]
dec.display_minor_ticks(True)
ra = ax1.coords[0]
ra.set_ticklabel()
ra.display_minor_ticks(True)
ax2 = plt.subplot(122,projection= muse_wcs,slices=('y', 'x'))
ax1.set_title('ALMA Data')
plt.colorbar(im1, ax=ax1)
# plt.colorbar(shrink = 0.7, label = 'log SFR Density [Kg yr^-1 kpc^-2')
# plt.title('Star Formation Rate Density')
im2 = ax2.imshow(logE_sfr_map,origin = 'lower')
dec = ax2.coords[1]
dec.display_minor_ticks(True)
ra = ax2.coords[0]
ra.set_ticklabel()
ra.display_minor_ticks(True)
ax2.set_title('MUSE Data')
plt.colorbar(im2, ax=ax2)
# plt.colorbar(shrink = 0.7, label = 'log Gas Surface density [Kg kpc^-2')
# plt.title('Gas Surface Density')
plt.show()

## Rebbining
x, y = np.meshgrid(np.arange(vel.shape[1]), np.arange(vel.shape[0]))
# xx, yy = np.meshgrid(np.arange(flux_H_alpha.shape[1]), np.arange(flux_H_alpha.shape[0]))

y_coordinates = [[t, 0] for t in range(450)]
yx_alma = alma_wcs.wcs_pix2world(y_coordinates, (0))
y_alma = [elem[0] for elem in yx_alma]
x_coordinates = [[0, t] for t in range(240)]
xy_alma = alma_wcs.wcs_pix2world(x_coordinates, (0))
x_alma = [elem[1] for elem in xy_alma]

ym_coordinates = [[t, 0] for t in range(318)]
yx_muse = muse_wcs.wcs_pix2world(ym_coordinates, (0))
y_muse = [elem[0] for elem in yx_muse]
xm_coordinates = [[0, t] for t in range(320)]
xy_muse = muse_wcs.wcs_pix2world(xm_coordinates, (0))
x_muse = [elem[1] for elem in xy_muse]

xx_alma, yy_alma = np.meshgrid(x_alma, y_alma)
xx_muse, yy_muse = np.meshgrid(x_muse, y_muse)

dpx = 0.1 
dpxx = 0.2 
# xx = xx - 25
# yy = yy - 10
# x_in = np.zeros([flux.shape[0] +1, flux.shape[1] + 1])
# y_in = np.zeros([flux.shape[0] +1, flux.shape[1] + 1])
# x_in[:-1, :-1] = xx_alma - dpx
# y_in[:-1, :-1] = yy_alma - dpx

# x_in[:-1, -1] = xx_alma[:,-1] + dpx
# x_in[-1,:] = x_in[-2,:]
# y_in[:-1, -1] = yy_alma[:,-1] + dpx
# y_in[-1,:] = y_in[-2,:]

# x_out = np.zeros([flux_H_alpha.shape[0] +1, flux_H_alpha.shape[1] + 1])
# y_out = np.zeros([flux_H_alpha.shape[0] +1, flux_H_alpha.shape[1] + 1])
# x_out[:-1, :-1] = xx_muse - dpxx
# y_out[:-1, :-1] = yy_muse - dpxx
# x_out[:-1, -1] = xx_muse[:,-1] + dpxx
# x_out[-1,:] = x_out[-2,:]
# y_out[:-1, -1] = yy_muse[:,-1] + dpxx
# y_out[-1,:] = y_out[-2,:]
alma_ny, alma_nx = vel.shape
alma_x0, alma_y0 = 3, -1
x_alma = np.arange(-alma_nx/2+ alma_x0,+alma_nx/2 + 0.5 + alma_x0, 1) * dpx
y_alma = np.arange(-alma_ny/2+ alma_y0,+alma_ny/2 + 0.5 + alma_y0, 1) * dpx

x_in, y_in  = np.meshgrid(x_alma, y_alma)

muse_ny, muse_nx = flux_H_alpha.shape
muse_x0, muse_y0 = 11, 8
x_muse = np.arange(-muse_nx/2+ muse_x0,+muse_nx/2 + 0.5 + muse_x0, 1) * dpxx
y_muse = np.arange(-muse_ny/2+ muse_y0,+muse_ny/2 + 0.5 + muse_y0, 1) * dpxx


x_out, y_out = np.meshgrid(x_muse, y_muse)

variance_in = np.full_like(x, 1)
quality_in = np.full_like(x, 1, dtype= np.uint64)
weights_in = np.full_like(x, 1)
# logE1[np.isnan(logE1)] = 0
rebinned_flux_CO_08 = regrid2d(x_in, y_in, logE1 , variance_in, quality_in, 
                            weights_in, x_out, y_out, divide_by_pixel_area= False)


plt.figure(figsize = (14,9))
plt.imshow(rebinned_flux_CO_08[0], origin = 'lower', cmap = 'jet', vmin = 36, vmax = 41)
plt.colorbar(shrink = 0.7, label = 'log Gas Surface density [Kg kpc^-2]')
plt.title('Gas Surface Density - alpha = 0.8')
plt.show()

rebinned_flux_CO_12 = regrid2d(x_in, y_in, logE2 , variance_in, quality_in, 
                            weights_in, x_out, y_out, divide_by_pixel_area= False)


plt.figure(figsize = (14,9))
plt.imshow(rebinned_flux_CO_08[0], origin = 'lower', cmap = 'jet', vmin = 36, vmax = 41)
plt.colorbar(shrink = 0.7, label = 'log Gas Surface density [Kg kpc^-2]')
plt.title('Gas Surface Density - alpha = 1.2')
plt.show()

rebinned_flux_CO_44 = regrid2d(x_in, y_in, logE3 , variance_in, quality_in, 
                            weights_in, x_out, y_out, divide_by_pixel_area= False)


plt.figure(figsize = (14,9))
plt.imshow(rebinned_flux_CO_08[0], origin = 'lower', cmap = 'jet', vmin = 36, vmax = 41)
plt.colorbar(shrink = 0.7, label = 'log Gas Surface density [Kg kpc^-2]')
plt.title('Gas Surface Density - alpha = 4.4')
plt.show()



logE_gas_map_08 = rebinned_flux_CO_08[0]
logE_gas_map_08[logE_gas_map_08==0] = np.nan
mask_gas_rebinned = ~np.isnan(logE_gas_map_08)
plt.figure(figsize = (14,9))
plt.imshow(logE_gas_map_08, origin = 'lower', cmap = 'jet', vmin = 36, vmax = 41)
plt.colorbar(shrink = 0.7, label = 'log Gas Surface density [Kg kpc^-2]')
plt.title('Gas Surface Density -alpha = 0.8 ')
plt.show()

logE_gas_map_12 = rebinned_flux_CO_12[0]
logE_gas_map_12[logE_gas_map_12==0] = np.nan
mask_gas_rebinned = ~np.isnan(logE_gas_map_12)
plt.figure(figsize = (14,9))
plt.imshow(logE_gas_map_12, origin = 'lower', cmap = 'jet', vmin = 36, vmax = 41)
plt.colorbar(shrink = 0.7, label = 'log Gas Surface density [Kg kpc^-2]')
plt.title('Gas Surface Density -alpha = 1.2 ')
plt.show()

logE_gas_map_44 = rebinned_flux_CO_44[0]
logE_gas_map_44[logE_gas_map_44==0] = np.nan
mask_gas_rebinned = ~np.isnan(logE_gas_map_44)
plt.figure(figsize = (14,9))
plt.imshow(logE_gas_map_44, origin = 'lower', cmap = 'jet', vmin = 36, vmax = 41)
plt.colorbar(shrink = 0.7, label = 'log Gas Surface density [Kg kpc^-2]')
plt.title('Gas Surface Density -alpha = 4.4 ')
plt.show()


logE_gas_map_08 = 1*logE_gas_map_08 - np.log10(K.M_sun.value) - 6 # M_sun pc^-2
logE_gas_map_12 = 1*logE_gas_map_12 - np.log10(K.M_sun.value) - 6 # M_sun pc^-2
logE_gas_map_44 = 1*logE_gas_map_44 - np.log10(K.M_sun.value) - 6 # M_sun pc^-2
logE_sfr_map = 1*logE_sfr_map - np.log10(K.M_sun.value)     # M_sun kpc^-2 yr^-1
# test2 = 10**(logE_gas_map) 
# test3 = np.nansum(test2)
# test4 = np.log10(test3)
# logE_gas = np.nansum(logE_gas_map)
logE_gas_08 = 1* logE_gas_tot_08 - np.log10(K.M_sun.value)  # M_sun pc^-2
logE_gas_12 = 1* logE_gas_tot_12 - np.log10(K.M_sun.value)  # M_sun pc^-2
logE_gas_44 = 1* logE_gas_tot_44 - np.log10(K.M_sun.value)  # M_sun pc^-2
logE_sfr = 1 * logE_sfr - np.log10(K.M_sun.value)     # M_sun kpc^-2 yr^-1


kennicut_08 = logE_sfr/ logE_gas_08
print('alpha = 0.8 logE_sfr / logE_gas = ')
print(kennicut_08)
kennicut_12 = logE_sfr/ logE_gas_12
print('alpha = 1.2 logE_sfr / logE_gas = ')
print(kennicut_12)
kennicut_44 = logE_sfr/ logE_gas_44
print('alpha = 4.4 logE_sfr / logE_gas = ')
print(kennicut_44)

kennicut_map_08 = logE_sfr_map / logE_gas_map_08
plt.figure(figsize = (14,9))
plt.imshow(kennicut_map_08, origin = 'lower', cmap = 'jet', vmin = -2, vmax= 2)
plt.colorbar(shrink = 0.7, label = 'log SFR Surface Density [M_sun kpc^-2 yr^-1] / log Gas Surface density [M_sun pc^-2]  ')
plt.title('Schmidt-Kennicut Map - Alpha = 0.8')
plt.show()

kennicut_map_12 = logE_sfr_map / logE_gas_map_12
plt.figure(figsize = (14,9))
plt.imshow(kennicut_map_12, origin = 'lower', cmap = 'jet', vmin = -2, vmax= 2)
plt.colorbar(shrink = 0.7, label = 'log SFR Surface Density [M_sun kpc^-2 yr^-1] / log Gas Surface density [M_sun pc^-2]  ')
plt.title('Schmidt-Kennicut Map - Alpha = 1.2')
plt.show()

kennicut_map_44 = logE_sfr_map / logE_gas_map_44
plt.figure(figsize = (14,9))
plt.imshow(kennicut_map_44, origin = 'lower', cmap = 'jet', vmin = -2, vmax= 2)
plt.colorbar(shrink = 0.7, label = 'log SFR Surface Density [M_sun kpc^-2 yr^-1] / log Gas Surface density [M_sun pc^-2]  ')
plt.title('Schmidt-Kennicut Map - Alpha = 4.4')
plt.show()

ics = np.linspace(-2,7, 1000)
ips1 = ics * 1.4 + np.log10(2.5*1e-4)
ips2 = ics * 1.55 + np.log10(2.5*1e-4)
ips3 = ics * 1.25 + np.log10(2.5*1e-4)
ips4 = ics * 0.66 - 2.70
plt.figure(figsize = (14,10))
plt.plot(ics, ips1, '-', label = 'n = 1.4' ) 
plt.plot(ics, ips2, '-',  label = 'n = 1.55'  ) 
plt.plot(ics, ips3, '-',  label = 'n = 1.25'  ) 
plt.scatter(logE_gas_08, logE_sfr, label = 'NGC6810')
plt.ylabel('log SFR Surface Density [M_sun kpc^-2 yr^-1]')
plt.xlabel('log Gas Surface density [M_sun pc^-2]')
plt.title('Schmidt-Kennicut Law - Alpha = 0.8')
plt.legend()
plt.show()

plt.figure(figsize = (14,10))
plt.plot(ics, ips1, '-', label = 'n = 1.4' ) 
plt.plot(ics, ips2, '-',  label = 'n = 1.55'  ) 
plt.plot(ics, ips3, '-',  label = 'n = 1.25'  ) 
plt.scatter(logE_gas_12, logE_sfr, label = 'NGC6810')
plt.ylabel('log SFR Surface Density [M_sun kpc^-2 yr^-1]')
plt.xlabel('log Gas Surface density [M_sun pc^-2]')
plt.title('Schmidt-Kennicut Law - Alpha = 1.2')
plt.legend()
plt.show()

plt.figure(figsize = (14,10))
plt.plot(ics, ips1, '-', label = 'n = 1.4' ) 
plt.plot(ics, ips2, '-',  label = 'n = 1.55'  ) 
plt.plot(ics, ips3, '-',  label = 'n = 1.25'  ) 
plt.scatter(logE_gas_44, logE_sfr, label = 'NGC6810')
plt.ylabel('log SFR Surface Density [M_sun kpc^-2 yr^-1]')
plt.xlabel('log Gas Surface density [M_sun pc^-2]')
plt.title('Schmidt-Kennicut Law - Alpha = 4.4')
plt.legend()
plt.show()



# (y, x) alma (450,240)
# (y, x) Muse (318, 320)



plt.figure(figsize = (14,10))
plt.scatter(logE_gas_map_08, logE_sfr_map, marker = '.')
plt.scatter(logE_gas_08, logE_sfr, marker='o')
plt.plot(ics, ips1 , label = 'n = 1.4' ) 
plt.plot(ics, ips2 ,  label = 'n = 1.55'  ) 
plt.plot(ics, ips3,  label = 'n = 1.25'  ) 
plt.plot(ics, ips4, label = 'nuova')
plt.title('Schmidt-Kennicut Law - Alpha = 0.8')
plt.legend()
plt.ylabel('log SFR Surface Density [M_sun kpc^-2 yr^-1]')
plt.xlabel('log Gas Surface density [M_sun pc^-2]')
# plt.xlim(-2,6)
# plt.ylim(-2, 5)
plt.show()


plt.figure(figsize = (14,10))
plt.scatter(logE_gas_map_12, logE_sfr_map, marker = '.')
plt.scatter(logE_gas_12, logE_sfr, marker='o')
plt.plot(ics, ips1 , label = 'n = 1.4' ) 
plt.plot(ics, ips2 ,  label = 'n = 1.55'  ) 
plt.plot(ics, ips3,  label = 'n = 1.25'  ) 
plt.plot(ics, ips4, label = 'nuova')
plt.title('Schmidt-Kennicut Law - Alpha = 1.2')
plt.legend()
plt.ylabel('log SFR Surface Density [M_sun kpc^-2 yr^-1]')
plt.xlabel('log Gas Surface density [M_sun pc^-2]')
# plt.xlim(-2,6)
# plt.ylim(-2, 5)
plt.show()

plt.figure(figsize = (14,10))
plt.scatter(logE_gas_map_44, logE_sfr_map, marker = '.')
plt.scatter(logE_gas_44, logE_sfr, marker='o')
plt.plot(ics, ips1 , label = 'n = 1.4' ) 
plt.plot(ics, ips2 ,  label = 'n = 1.55'  ) 
plt.plot(ics, ips3,  label = 'n = 1.25'  ) 
plt.plot(ics, ips4, label = 'nuova')
plt.title('Schmidt-Kennicut Law - Alpha = 4.4')
plt.legend()
plt.ylabel('log SFR Surface Density [M_sun kpc^-2 yr^-1]')
plt.xlabel('log Gas Surface density [M_sun pc^-2]')
# plt.xlim(-2,6)
# plt.ylim(-2, 5)
plt.show()


# Dividing the galaxy above/up the Kennicut_Schmidt law


# Alpha = 0.8
map_y_08 = np.copy(logE_gas_map_08)
eKS_08 =  map_y_08 * 0.66 -2.70
# mask_map_08 = np.ma.array(logE_sfr_map) 
cond08_1 = logE_sfr_map > eKS_08
cond08_2 = logE_sfr_map < eKS_08
# map_up_08 = np.ma.masked_where(cond08_1  , mask_map_08)
# map_down_08 = np.ma.masked_where(cond08_2, mask_map_08)


map_up_08 = 1.0 * np.ones_like(logE_sfr_map)
map_up_08[cond08_1 == False] = np.nan
map_down_08 = 1.0 * np.ones_like(logE_sfr_map)
map_down_08[cond08_2 == False] = np.nan

#Plot

plt.figure(figsize = (14,9))

plt.imshow(map_up_08, origin = 'lower', cmap = 'Reds',interpolation = 'none')
plt.colorbar(shrink = 0.7, label = '> kennicut-schmidt law')
plt.imshow(map_down_08, origin = 'lower', cmap ='Blues',interpolation = 'none')
plt.colorbar(shrink = 0.7, label='< > kennicut-schmidt law')
plt.title('Kennicut-Schmidt law map with alpha = 0.8')
plt.show()



# Alpha = 1.2
map_y_12 = np.copy(logE_gas_map_12)
eKS_12 =  map_y_12 * 0.66 -2.70
# mask_map_12 = np.ma.array(kennicut_map_12)
cond12_1 = logE_sfr_map > eKS_12
cond12_2 = logE_sfr_map < eKS_12
# map_up = np.ma.masked_where(cond12_1  , mask_map_12)
# map_down = np.ma.masked_where(cond12_2, mask_map_12)


map_up_12 = 1.0 * np.ones_like(logE_sfr_map)
map_up_12[cond12_1 == False] = np.nan
map_down_12 = 1.0 * np.ones_like(logE_sfr_map)
map_down_12[cond12_2 == False] = np.nan

#Plot

plt.figure(figsize = (14,9))

plt.imshow(map_up_12, origin = 'lower', cmap = 'Reds',interpolation = 'none')
plt.colorbar(shrink = 0.7, label = '> kennicut-schmidt law')
plt.imshow(map_down_12, origin = 'lower', cmap ='Blues',interpolation = 'none')
plt.colorbar(shrink = 0.7, label='< > kennicut-schmidt law')
plt.title('Kennicut-Schmidt law map with alpha = 1.2')
plt.show()




# Alpha = 4.4
map_y_44 = np.copy(logE_gas_map_44)
eKS_44 =  map_y_44 * 0.66 -2.70
# mask_map_44 = np.ma.array(logE_sfr_map)
cond44_1 = logE_sfr_map > eKS_44
cond44_2 = logE_sfr_map < eKS_44
# map_up = np.ma.masked_where(cond44_1  , mask_map)
# map_down = np.ma.masked_where(cond44_2, mask_map)


map_up_44 = 1.0 * np.ones_like(kennicut_map_44)
map_up_44[cond44_1 == False] = np.nan
map_down_44 = 1.0 * np.ones_like(kennicut_map_44)
map_down_44[cond44_2 == False] = np.nan

#Plot

plt.figure(figsize = (14,9))

plt.imshow(map_up_44, origin = 'lower', cmap = 'Reds',interpolation = 'none')
plt.colorbar(shrink = 0.7, label = '> kennicut-schmidt law')
plt.imshow(map_down_44, origin = 'lower', cmap ='Blues',interpolation = 'none')
plt.colorbar(shrink = 0.7, label='< kennicut-schmidt law')
plt.title('Kennicut-Schmidt law map with alpha = 4.4')
plt.show()