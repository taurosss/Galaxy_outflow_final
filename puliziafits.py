#import python libraries
import numpy as np 
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.wcs as wcs
import os 
import time
#from astropy.nddata import Cutout2D
from scipy import ndimage
import astropy.constants as K
import astropy.units as u
from astropy.cosmology import Planck15 as p15
import scipy.ndimage
from lmfit import minimize, Parameters, report_fit



def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


path = os.path.dirname(os.path.abspath('__file__'))

flux = fits.open(path + '/moments_map/NGC6810/flux_map_spiral_5_2gx.fits')[0]
vel = fits.open(path + '/moments_map/NGC6810/vel_map_spiral_5_2gx.fits')[0]
disp = fits.open(path + '/moments_map/NGC6810/vdisp_map_spiral_5_2gx.fits')[0]

plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(flux.data, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7)
plt.subplot(132)
plt.imshow(vel.data, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar(shrink = 0.7)
plt.subplot(133)
plt.imshow(disp.data, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar(shrink = 0.7)

Ny,Nx = 500, 500
flux1 = flux.data
vel1 = vel.data
disp1 = disp.data

yy, xx = np.mgrid[:Ny, :Nx]
x0, y0 = 250, 250

yy = yy -y0
xx = xx - x0

vel2 = np.resize(vel1,(500,500))


# test = rebin(xx, (167,167))
# test2 = rebin1(xx, (30,30))

# flux1 = flux.data
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj < 1.22 * ii - 220:
#             flux1[jj,ii] = np.nan

# vel1 = vel.data
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj < 1.22 * ii - 220:
#             vel1[jj,ii] = np.nan
    
# disp1 = disp.data
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj < 1.22 * ii - 220:
#             disp1[jj,ii] = np.nan
            
# flux2 = flux1
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj > 1.014 * ii +279:
#             flux2[jj,ii] = np.nan

# vel2 = vel1
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj > 1.014 * ii +279:
#             vel2[jj,ii] = np.nan
    
# disp2 = disp1
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj > 1.014 * ii +279:
#             disp2[jj,ii] = np.nan

# plt.figure(figsize = (12,4))

# plt.subplot(131)
# plt.imshow(flux2, origin = 'lower', cmap = 'jet')
# plt.colorbar(shrink = 0.7)
# plt.subplot(132)
# plt.imshow(vel2, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
# plt.colorbar(shrink = 0.7)
# plt.subplot(133)
# plt.imshow(disp2, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
# plt.colorbar(shrink = 0.7)

# flux3 = flux2
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj < -1.515 * ii +294:
#             flux3[jj,ii] = np.nan

# vel3 = vel2
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj <-1.515 * ii +294:
#             vel3[jj,ii] = np.nan
    
# disp3 = disp2
# for jj in range (Ny):
#     for ii in range(Nx):
#         if jj <-1.515 * ii +294:
#             disp3[jj,ii] = np.nan
            
            
# plt.figure(figsize = (12,4))

# plt.subplot(131)
# plt.imshow(flux3, origin = 'lower', cmap = 'jet')
# plt.colorbar(shrink = 0.7)
# plt.subplot(132)
# plt.imshow(vel3, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
# plt.colorbar(shrink = 0.7)
# plt.subplot(133)
# plt.imshow(disp3, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
# plt.colorbar(shrink = 0.7)

# hdu = fits.PrimaryHDU(flux3)
# hdul = fits.HDUList([hdu])
# hdul.writeto('flux3.fits')
# hdu = fits.PrimaryHDU(vel3)
# hdul = fits.HDUList([hdu])
# hdul.writeto('vel3.fits')
# hdu = fits.PrimaryHDU(disp3)
# hdul = fits.HDUList([hdu])
# hdul.writeto('vdisp3.fits')
