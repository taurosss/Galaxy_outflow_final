#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tauro
"""

#import python libraries
import numpy as np 
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os 





# def congrid(a, newdims, method='linear', centre=False, minusone=False):
#     '''Arbitrary resampling of source array to new dimension sizes.
#     Currently only supports maintaining the same number of dimensions.
#     To use 1-D arrays, first promote them to shape (x,1).
    
#     Uses the same parameters and creates the same co-ordinate lookup points
#     as IDL''s congrid routine, which apparently originally came from a VAX/VMS
#     routine of the same name.

#     method:
#     neighbour - closest value from original data
#     nearest and linear - uses n x 1-D interpolations using
#                          scipy.interpolate.interp1d
#     (see Numerical Recipes for validity of use of n 1-D interpolations)
#     spline - uses ndimage.map_coordinates

#     centre:
#     True - interpolation points are at the centres of the bins
#     False - points are at the front edge of the bin

#     minusone:
#     For example- inarray.shape = (i,j) & new dimensions = (x,y)
#     False - inarray is resampled by factors of (i/x) * (j/y)
#     True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
#     This prevents extrapolation one element beyond bounds of input array.
#     '''
#     if not a.dtype in [np.float64, np.float32]:
#         a = np.cast[float](a)

#     m1 = np.cast[int](minusone)
#     ofs = np.cast[int](centre) * 0.5
#     old = np.array( a.shape )
#     ndims = len( a.shape )
#     if len( newdims ) != ndims:
#         print ("[congrid] dimensions error. " + "This routine currently only support "+"rebinning to the same number of dimensions.")
#         return None
#     newdims = np.asarray( newdims, dtype=float )
#     dimlist = []

#     if method == 'neighbour':
#         for i in range( ndims ):
#             base = np.indices(newdims)[i]
#             dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
#                             * (base + ofs) - ofs )
#         cd = np.array( dimlist ).round().astype(int)
#         newa = a[list( cd )]
#         return newa

#     elif method in ['nearest','linear']:
#         # calculate new dims
#         for i in range( ndims ):
#             base = np.arange( newdims[i] )
#             dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
#                             * (base + ofs) - ofs )
#         # specify old dims
#         olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

#         # first interpolation - for ndims = any
#         mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
#         newa = mint( dimlist[-1] )

#         trorder = [ndims - 1] + range( ndims - 1 )
#         for i in range( ndims - 2, -1, -1 ):
#             newa = newa.transpose( trorder )

#             mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
#             newa = mint( dimlist[i] )

#         if ndims > 1:
#             # need one more transpose to return to original dimensions
#             newa = newa.transpose( trorder )

#         return newa
#     elif method in ['spline']:
#         oslices = [ slice(0,j) for j in old ]
#         oldcoords = np.ogrid[oslices]
#         nslices = [ slice(0,j) for j in list(newdims) ]
#         newcoords = np.mgrid[nslices]

#         newcoords_dims = range(np.rank(newcoords))
#         #make first index last
#         newcoords_dims.append(newcoords_dims.pop(0))
#         newcoords_tr = newcoords.transpose(newcoords_dims)
#         # makes a view that affects newcoords

#         newcoords_tr += ofs

#         deltas = (np.asarray(old) - m1) / (newdims - m1)
#         newcoords_tr *= deltas

#         newcoords_tr -= ofs

#         newa = scipy.ndimage.map_coordinates(a, newcoords)
#         return newa
#     else:
#         print ("Congrid error: Unrecognized interpolation type.\n", \
#               "Currently only \'neighbour\', \'nearest\',\'linear\',", \
#               "and \'spline\' are supported.")
#         return None



def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
              new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


path = os.path.dirname(os.path.abspath('__file__'))

flux = fits.open(path + '/moments_map/NGC6810/flux_map_spiral_6_2gxNGC6810.fits')[0]
vel = fits.open(path + '/moments_map/NGC6810/vel_crop.fits')[0]
disp = fits.open(path + '/moments_map/NGC6810/vdisp_crop.fits')[0]
velerr = fits.open(path + '/moments_map/NGC6810/vel_err_crop.fits')[0]
disperr = fits.open(path + '/moments_map/NGC6810/vdisp_err_crop.fits')[0]

# plt.figure(figsize=(12,4))

# plt.subplot(131)
# plt.imshow(flux.data, origin = 'lower', cmap = 'jet')
# plt.colorbar(shrink = 0.7)
# plt.subplot(132)
# plt.imshow(vel.data, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
# plt.colorbar(shrink = 0.7)
# plt.subplot(133)
# plt.imshow(disp.data, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
# plt.colorbar(shrink = 0.7)



vel = vel.data
disp = disp.data
velerr = velerr.data
disperr = disperr.data




# vel1 = rebin(vel,(225,120))
# disp1 = rebin(disp,(225,120))
# velerr1 = rebin(velerr, (225,120))
# disperr1 = rebin(disperr, (225,120)) 

# plt.figure(figsize=(12,4))

# plt.subplot(121)
# plt.imshow(vel1, origin = 'lower', cmap = 'jet')
# plt.colorbar(shrink = 0.7)
# plt.subplot(122)
# plt.imshow(disp1, origin = 'lower', vmin = 0, vmax = 100, cmap ='jet')
# plt.colorbar(shrink = 0.7)


Ny,Nx = 450, 240
yy, xx = np.mgrid[:Ny, :Nx]

x0, y0 = 120, 225

yy = yy -y0
xx = xx - x0


A = xx.flatten()
B = yy.flatten()
C = vel.flatten()
D = velerr.flatten()
E = disp.flatten()
F = disperr.flatten()
# A = A /60
# B = B/120


AA = np.delete(A, np.isnan(C))
BB = np.delete(B, np.isnan(C))
CC = np.delete(C, np.isnan(C))
DD = np.full_like(CC, np.abs(CC*0.01))
# DD = np.delete(D, np.isnan(C))
EE = np.delete(E, np.isnan(C))
FF = np.full_like(EE, np.abs(EE*0.01))
# FF = np.delete(E, np.isnan(C))


#togliere i nan con np.delete
DataOut = np.column_stack((AA,BB,CC,DD,EE,FF))
np.savetxt('NGC6810_240_450.dat', DataOut, fmt='%.4f')
        
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
