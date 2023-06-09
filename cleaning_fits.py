#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:11:11 2023

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
import plotbin as pb
from plotbin import symmetrize_velfield
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from math import degrees, radians
from scipy.signal import savgol_filter
from plotbin.sauron_colormap import register_sauron_colormap


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

flux = fits.open(path + '/moments_map/NGC6810/flux_crop.fits')[0]  #già moltiplicato per dv
vel = fits.open(path + '/moments_map/NGC6810/vel_crop.fits')[0]
disp = fits.open(path + '/moments_map/NGC6810/vdisp_crop.fits')[0]
velerr = fits.open(path + '/moments_map/NGC6810/vel_err_crop.fits')[0]
disperr = fits.open(path + '/moments_map/NGC6810/vdisp_err_crop.fits')[0]
fluxerr = fits.open(path + '/moments_map/NGC6810/flux_err_crop.fits')[0]

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



# Define the corner points of the triangle region
x1, y1 = 0, 300
x2, y2 = 100, 450
x3, y3 = 0, 450

# Define the equation for the lines that define the triangle
# Using the point-slope form of the line equation y = mx + b
m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1


# Set values in the triangle region to NaN
for j in range(450):
    for i in range(240):
        y = j
        x = i
        if y > m * x + b:
            disp[j,i] = np.nan
            vel[j,i] = np.nan
            flux[j,i] = np.nan
            velerr[j,i] = np.nan
            disperr[j,i] = np.nan
            fluxerr[j,i] = np.nan
            
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(flux, origin = 'lower', cmap = 'jet')
plt.colorbar()
plt.subplot(132)
plt.imshow(vel, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar()
plt.subplot(133)
plt.imshow(disp, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar()

# Define the corner points of the triangle region
x1, y1 = 70, 0
x2, y2 = 0, 100
x3, y3 = 0, 450

# Define the equation for the lines that define the triangle
# Using the point-slope form of the line equation y = mx + b
m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1


# Set values in the triangle region to NaN
for j in range(450):
    for i in range(240):
        y = j
        x = i
        if y < m * x + b:
            disp[j,i] = np.nan
            vel[j,i] = np.nan
            flux[j,i] = np.nan
            velerr[j,i] = np.nan
            disperr[j,i] = np.nan
            fluxerr[j,i] = np.nan
            
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(flux, origin = 'lower', cmap = 'jet')
plt.colorbar()
plt.subplot(132)
plt.imshow(vel, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar()
plt.subplot(133)
plt.imshow(disp, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar()

# Define the corner points of the triangle region
x1, y1 = 150, 0
x2, y2 = 225, 75
x3, y3 = 0, 450

# Define the equation for the lines that define the triangle
# Using the point-slope form of the line equation y = mx + b
m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1


# Set values in the triangle region to NaN
for j in range(450):
    for i in range(240):
        y = j
        x = i
        if y < m * x + b:
            disp[j,i] = np.nan
            vel[j,i] = np.nan
            flux[j,i] = np.nan
            velerr[j,i] = np.nan
            disperr[j,i] = np.nan
            fluxerr[j,i] = np.nan
            
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(flux, origin = 'lower', cmap = 'jet')
plt.colorbar()
plt.subplot(132)
plt.imshow(vel, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar()
plt.subplot(133)
plt.imshow(disp, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar()


# Define the corner points of the triangle region
x1, y1 = 225, 400
x2, y2 = 120, 450
x3, y3 = 0, 450

# Define the equation for the lines that define the triangle
# Using the point-slope form of the line equation y = mx + b
m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1


# Set values in the triangle region to NaN
for j in range(450):
    for i in range(240):
        y = j
        x = i
        if y > m * x + b:
            disp[j,i] = np.nan
            vel[j,i] = np.nan
            flux[j,i] = np.nan
            
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(flux, origin = 'lower', cmap = 'jet')
plt.colorbar()
plt.subplot(132)
plt.imshow(vel, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar()
plt.subplot(133)
plt.imshow(disp, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar()


hdu = fits.PrimaryHDU(flux)
hdul = fits.HDUList([hdu])
hdul.writeto('flux_crop_clean.fits', overwrite = True)
hdu = fits.PrimaryHDU(vel)
hdul = fits.HDUList([hdu])
hdul.writeto('vel_crop_clean.fits', overwrite = True)
hdu = fits.PrimaryHDU(disp)
hdul = fits.HDUList([hdu])
hdul.writeto('vdisp_crop_clean.fits', overwrite = True)
hdu = fits.PrimaryHDU(fluxerr)
hdul = fits.HDUList([hdu])
hdul.writeto('flux_err_crop_clean.fits', overwrite = True)
hdu = fits.PrimaryHDU(velerr)
hdul = fits.HDUList([hdu])
hdul.writeto('vel_err_crop_clean.fits', overwrite = True)
hdu = fits.PrimaryHDU(disperr)
hdul = fits.HDUList([hdu])
hdul.writeto('vdisp_err_crop_clean.fits', overwrite = True)