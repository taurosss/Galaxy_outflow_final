#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tauro
"""

# import python libraries
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
from heapq import nlargest
import scipy.interpolate



#----------------------------------------------------------------------------

def residual(pars, x, p, data=None, sigma=None):
    ## Multi-gaussians model
    
    argu1 = (x - pars['cen_g1'])**2 / (2*(pars['wid_g1'])**2)

    if p == 1:
        model = pars['amp_g1'] * np.exp(-argu1) 
    if p == 2:
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2'] * np.exp(-argu2))
    if p == 3:
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3'] * np.exp(-argu3))
    if p == 4:
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        argu4 = (x - pars['cen_g4'])**2 / (2*(pars['wid_g4'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3']*np.exp(-argu3) + pars['amp_g4']*np.exp(-argu4))
    if p == 5:
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        argu4 = (x - pars['cen_g4'])**2 / (2*(pars['wid_g4'])**2)
        argu5 = (x - pars['cen_g5'])**2 / (2*(pars['wid_g5'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3']*np.exp(-argu3) + pars['amp_g4']*np.exp(-argu4) + pars['amp_g5']*np.exp(-argu5))
    
    if data is None:
            return model
    if sigma is None:
        return model - data
    return (model - data) / sigma

#----------------------------------------------------------------------------

##FUNCTIONS TO MAKE THE SPIRAL GRID FOR FITTING
def invers_spiral(A):
    return A[::-1]                 # inverting the array, so it starts from the center 

def spiral_mat_to_vect(A):
    v = []
    while(A.size != 0):
        v.append(A[0,:])
        A = A[1:,:].T[::-1]
    return np.concatenate(v)

def spiral_vect_to_mat(v):
    L = int(np.sqrt(v.size))       # lenght of the piece to add
    l = L
    A = np.zeros((L,L))
    i = 3                          # starting from 3, so in this way the x coordinate will increase on the second step 
    x = 0                          # x coordinate of the new piece
    y = 0                          # y coordinate of the new piece
    
    A[x,y:l] = v[0:l]
    A = A.T[::-1]
    v = v[l:len(v)]

    while(v.size != 0):
        i += 1                     # In every step: rotate and fill the first raw of the matrix
        if i % 2 == 0:             # Every 2 rotations l decreases
            l -= 1
        if (i + 1) % 4 == 0:       # Every 4 rotations x increases
            x += 1
        if i % 4 == 0:             # Every 4 rotations y increases with a delay of 1 step compared to x
            y += 1
        A[x,y:y+l] = v[0:l]
        A = A.T[::-1]
        v = v[l:len(v)]
        
    for rotations in range(i % 4): # The last rotations to have the matrix rotated in the correct way
        A = A.T[::-1]
        
    return A

#----------------------------------------------------------------------------    

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
print(frequency_mean)


# z = v/c = (nu_emit - nu_obs)/nu_obs 
velocity_unit = ((frequency_mean- (frequency*u.GHz))/(frequency*u.GHz))*K.c.to('km/s')
print(velocity_unit[:10])
velocity = velocity_unit.value
print(velocity[:10])
dv = velocity[0]-velocity[1]

#----------------------------------------------------------------------------

# TOTAL SPECTRUM
# location of the target
x0,y0 = 250, 250
# size of the square aperture 
dl = 100
# extract the spectrum
spectrum = np.nansum(datacube.data[:,y0-dl:y0+dl,x0-dl:x0+dl],axis = (1,2))
# 1plot: frequency - spectrum
plt.plot(frequency, spectrum, label = 'data')
plt.plot(frequency,frequency*0,':',color = 'black')
plt.xlabel('frequency [GHz]')
plt.ylabel('flux ')
plt.title('Total Spectrum')
plt.legend()
plt.show()



# 2plot: velocity - spectrum
plt.figure(figsize = (12,4))
plt.plot(velocity, spectrum, label = 'data')
plt.plot(velocity,frequency*0,':',color = 'black')
plt.xlabel('velocity [km/s]')
plt.ylabel('flux ')
plt.title('Totale Spectrum (function of velocity)')
plt.legend()
plt.show()

#----------------------------------------------------------------------------

## RMS DETERMINATION WITH THE POWER RESPONSE 

# data/power response
noise_cube = datacube.data / datacube_antenna.data

# Choosing an empty region
x0, y0 = 318, 163
dl = 20
noise = noise_cube[:,y0-dl:y0+dl,x0-dl:x0+dl]
error = np.std(noise[1:,:,:])

print("rms  = {:2f} mJy".format(error))
print("####################")

#----------------------------------------------------------------------------

# Fit and plot of the total spectrum
x = velocity
data = spectrum
p = 1

fit_params = Parameters()
fit_params.add('amp_g1', value=50,)
fit_params.add('cen_g1', value=0)
fit_params.add('wid_g1', value=100)

out = minimize(residual, fit_params, args=(x,p,), kws={'data': data})
fit = residual(out.params, x, p)
print('##')
print('1gaussian_fit')
report_fit(out)
parvals = out.params.valuesdict()
stddev_t = parvals['wid_g1']
print("FWHM  = {:2f} km/s".format(2.355*stddev_t))


bic_1g = out.bic            #Bayesian Crit Info for the fit with 1 Gaussian

plt.figure(figsize = (12,4))
plt.plot(x, data, label='data')
plt.plot(x, data*0,':',color = 'black')
plt.plot(x, fit, label='best fit')
plt.xlabel('velocity [km/s]')
plt.ylabel('flux [mJy]')
plt.title('1Gaussian_model')
plt.legend()
plt.show()

#----------------------------------------------------------------------------

# Generating moments map without fitting

#datacube.data = np.where(datacube.data<0, 0*datacube.data, datacube.data)
mask_cube = np.where(datacube.data > 2.5*error, datacube.data, np.nan)
M0 = np.nansum(mask_cube, axis = (0))*dv
M1 = np.nansum(mask_cube[:,:,:]*velocity[:,np.newaxis,np.newaxis], axis=0)*dv / M0
# avoid division by 0 or neg values in sqrt
# thr = np.nanpercentile(M0[np.where(M0>0)],0.01)
thr = 2.5*error
M0[np.where(M0<thr)]=np.nan
M2 = np.sqrt(np.nansum(np.power(datacube.data[:, :, :] * (velocity[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :]),2), axis=0) * dv / M0 )     


plt.figure(figsize = (12,4))
plt.suptitle('brute moment maps')
plt.subplot(131)
plt.imshow(M0, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7)
plt.subplot(132)
plt.imshow(M1, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar(shrink = 0.7)
plt.subplot(133)
plt.imshow(M2, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar(shrink = 0.7)


#----------------------------------------------------------------------------

# generete model maps 

mod = np.zeros_like(datacube.data)
mod1 = np.zeros_like(datacube.data)
mod2 = np.zeros_like(datacube.data)
modx = np.zeros_like(datacube.data)
mod1x = np.zeros_like(datacube.data)
mod2x= np.zeros_like(datacube.data)
# res = np.zeros_like(datacube.data)
# res1 = np.zeros_like(datacube.data)
# res2 = np.zeros_like(datacube.data)
# res3 = np.zeros_like(datacube.data)

#----------------------------------------------------------------------------

# Creating the matrix-coordinates that would be spiralized
mat_tmp =  np.empty([Nx, Ny], dtype='<U7')
for i in range(Nx):
    for j in range(Ny):
        mat_tmp[i,j] = (str(j) + '-' + str(i))  
        
#Spiralizing the matrix
spiral_vect = spiral_mat_to_vect(mat_tmp)
spiral_vect = invers_spiral(spiral_vect)

#----------------------------------------------------------------------------

#Generating the moments map and other stuff
flux_map_tmp = np.nansum(datacube.data, axis = (0))
flux_map = np.full_like(flux_map_tmp, np.nan)
vel_map = np.full_like(flux_map, np.nan)
vdisp_map = np.full_like(flux_map, np.nan)
amp1_1err= np.full_like(flux_map, np.nan)
cen1_1err= np.full_like(amp1_1err, np.nan)
wid1_1err= np.full_like(amp1_1err, np.nan)
amp2_1err =np.full_like(amp1_1err, np.nan)
amp2_2err =np.full_like(amp1_1err, np.nan)
cen2_1err =np.full_like(amp1_1err, np.nan)
cen2_2err =np.full_like(amp1_1err, np.nan)
wid2_1err =np.full_like(amp1_1err, np.nan)
wid2_2err=np.full_like(amp1_1err, np.nan)
amp1x_1err=np.full_like(amp1_1err, np.nan)
cen1x_1err=np.full_like(amp1_1err, np.nan)
wid1x_1err=np.full_like(amp1_1err, np.nan)
amp2x_1err=np.full_like(amp1_1err, np.nan)
amp2x_2err=np.full_like(amp1_1err, np.nan)
cen2x_1err=np.full_like(amp1_1err, np.nan)
cen2x_2err=np.full_like(amp1_1err, np.nan)
wid2x_1err=np.full_like(amp1_1err, np.nan)
wid2x_2err=np.full_like(amp1_1err, np.nan)
amperr = np.full_like(amp1_1err, np.nan)
velerr = np.full_like(amp1_1err, np.nan)
disperr = np.full_like(amp1_1err, np.nan) 
#velarr = np.full_like(amp1_1err, np.nan) 
#disparr = np.full_like(amp1_1err, np.nan) 

#----------------------------------------------------------------------------

#Initialize the initial parameters for the lmfit of the first pixel (the center)
fit_params1g = Parameters()
fit_params1g.add('amp_g1', value=0.035, min = 0.005, max= 0.1)
fit_params1g.add('cen_g1', value=M1[250,250], min = -300, max= 300)
fit_params1g.add('wid_g1', value=150, min = 10, max = 200)

fit_params2g = Parameters()
fit_params2g.add('amp_g1', value=0.035, min = 0.005, max= 0.1)
fit_params2g.add('cen_g1', value=M1[250,250], min = -300, max= 300)
fit_params2g.add('wid_g1', value=60, min = 10, max = 200)
fit_params2g.add('amp_g2', value=0.02, min = 0.005, max= 0.1)
fit_params2g.add(name='peak_split', value=50, min=-200, max=250, vary=True)
fit_params2g.add(name=('cen_g2'), expr='peak_split+cen_g1')
fit_params2g.add('wid_g2', value=60, min = 10, max = 200)

fit_params1gx = Parameters()
fit_params1gx.add('amp_g1', value=0.035, min = 0.005, max= 0.1)
fit_params1gx.add('cen_g1', value=M1[250,250], min = -300, max= 300)
fit_params1gx.add('wid_g1', value=150, min = 10, max = 200)

fit_params2gx = Parameters()
fit_params2gx.add('amp_g1', value=0.035, min = 0.005, max= 0.1)
fit_params2gx.add('cen_g1', value=M1[250,250], min = -300, max= 300)
fit_params2gx.add('wid_g1', value=60, min = 10, max = 200)
fit_params2gx.add('amp_g2', value=0.02, min = 0.005, max= 0.1)
fit_params2gx.add(name='peak_split', value=25, min=-200, max=250, vary=True)
fit_params2gx.add(name=('cen_g2'), expr='peak_split+cen_g1')
fit_params2gx.add('wid_g2', value=60, min = 10, max = 200)

#----------------------------------------------------------------------------

# velmin, velmax = -300, 300
range1 = list(range(31,480))        # y range for the final maps  
range2 = list(range(106,345))       # x range for the final maps
printplot = True
printfit = True

#----------------------------------------------------------------------------

# Spiral Fit
for idxs in spiral_vect[0:2]:
    prt = idxs.partition("-")                       # Take the coordinates of the pixel in the spiral fitting
    ii = int(prt[2])    #x
    jj = int(prt[0])    #y
    spec_tmp = datacube.data[:,jj,ii]
    spec_tmp = np.nan_to_num(spec_tmp)
    spec_tmp[0]=0
    if all(nlargest(2, spec_tmp) > 3.5*error):      # thershold on the 2 biggest values of the spectra
    # if flux_map_tmp[jj,ii]>4.2*error:
    #     spec_tmp = datacube.data[:,jj,ii]
    #     spec_tmp = np.nan_to_num(spec_tmp)
    #     spec_tmp[0]=0
        
        ##FIT WITH 1 GAUSSIAN
        if jj > 1.14777 * ii + 20:                  # imposing differnt limit on the pixel above the galaxy diagonal
            velmax = 90                             # limit to select the blueshifted pixel
            velmin = -300
        else:
            velmax = 300
            velmin=-90
        p=1
        # fit_params1g.add('cen_g1', value=M1[jj,ii], min = velmin, max= velmax)
        out = minimize(residual, fit_params1g, args=(x,p,), kws={'data': spec_tmp,'sigma': error/2.5})
        fit1 = residual(out.params, x, p)
        mod1[:,jj,ii] = fit1
        parvals = out.params.valuesdict()
        amplitude = parvals['amp_g1']
        stddev = parvals['wid_g1']
        mean = parvals['cen_g1']
        amp1_1err[jj,ii] = out.params['amp_g1'].stderr
        cen1_1err[jj,ii]= out.params['cen_g1'].stderr
        wid1_1err[jj,ii] = out.params['wid_g1'].stderr
        out_1g = [amplitude, mean, stddev]
        bic_1g = out.bic
        # chi1 = out.redchi
        
        # Pass the best fit parameters as initial parameters for the next pixel to fit
        fit_params1g.add('amp_g1', value=out_1g[0], min = 0.0025, max= 0.1)
        fit_params1g.add('cen_g1', value=out_1g[1], min = velmin, max= velmax)
        fit_params1g.add('wid_g1', value=out_1g[2], min = 1, max = 300)
        
        if printplot == True:
            plt.figure(figsize = (12,4))
            plt.plot(x, spec_tmp, label='data')
            plt.plot(x, data*0,':',color = 'black')
            plt.plot(x, fit1, label='best fit')
            plt.xlabel('velocity [km/s]')
            plt.ylabel('flux [mJy]')
            plt.title('1Gaussian_model_pixel' + str(ii) + '-' + str(jj) )
            plt.legend()
            plt.show()
        if printfit == True:
            print('##')
            print('1gaussian_fit pixel:' + str(ii) + '-' + str(jj))
            report_fit(out)
            
        ##FIT WITH 2 GAUSSIANS
        p = 2
        n = 2
        # fit_params2g.add('cen_g1', value=M1[jj,ii], min = velmin, max= velmax)
        out = minimize(residual, fit_params2g, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/2.5})
        fit2 = residual(out.params, x, p)
        # res2[:,jj,ii] = residual(out.params, x, p, data, error/3) 
        mod2[:,jj,ii] = fit2
        parvals = out.params.valuesdict()
        amplitude_2 = parvals['amp_g' + str(n)]
        stddev_2 = parvals['wid_g' + str(n)]
        mean_2 = parvals['cen_g' + str(n)]
        amplitude_1 = parvals['amp_g' + str(n-1)]
        stddev_1 = parvals['wid_g' + str(n-1)]
        mean_1 = parvals['cen_g' + str(n-1)]
        amp2_1err[jj,ii] = out.params['amp_g1'].stderr
        cen2_1err[jj,ii]= out.params['cen_g1'].stderr
        wid2_1err[jj,ii] = out.params['wid_g1'].stderr
        amp2_2err[jj,ii] = out.params['amp_g2'].stderr
        cen2_2err[jj,ii]= out.params['cen_g2'].stderr
        wid2_2err[jj,ii] = out.params['wid_g2'].stderr
        out_2g = [amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2]
        bic_2g = out.bic
        # chi2 = out.redchi
        
        # Pass the best fit parameters as initial parameters for the next pixel to fit
        fit_params2g.add('amp_g1', value=out_2g[0], min = 0.0025, max= 0.1)
        fit_params2g.add('cen_g1', value=out_2g[1], min = velmin, max= velmax)
        fit_params2g.add('wid_g1', value=out_2g[2], min = 1, max = 200)
        fit_params2g.add('amp_g2' , value=out_2g[3], min= 0.0025, max= 0.1)
        fit_params2g.add(name=('cen_g2'), expr='peak_split+cen_g1')
        fit_params2g.add('wid_g2', value=out_2g[5], min =1, max= 200)
        
        if printplot == True:
            plt.figure(figsize = (12,4))
            plt.plot(x, spec_tmp, label='data')
            plt.plot(x, data*0,':',color = 'black')
            plt.plot(x, fit2, label='best fit')
            plt.xlabel('velocity [km/s]')
            plt.ylabel('flux [mJy]')
            plt.title('2Gaussian_model_pixel' + str(ii) + '-' + str(jj) )
            plt.legend()
            plt.show()
        if printfit == True:
            print('##')
            print(str(n) + 'gaussianfit' + 'pixel:' + str(ii) + '-' + str(jj))
            report_fit(out)
        
        ##FIT WITH 1 GAUSSIAN with initial velocity taken from the "brute" moment 1
        p=1
        fit_params1gx.add('cen_g1', value=M1[jj,ii], min = velmin, max= velmax)
        fit_params1gx.add('wid_g1', value=M2[jj,ii], min = 1, max = 300)
        out = minimize(residual, fit_params1gx, args=(x,p,), kws={'data': spec_tmp,'sigma': error/2.5})
        fit1x = residual(out.params, x, p)
        mod1x[:,jj,ii] = fit1x
        parvals = out.params.valuesdict()
        amplitude = parvals['amp_g1']
        stddev = parvals['wid_g1']
        mean = parvals['cen_g1']
        amp1x_1err[jj,ii] = out.params['amp_g1'].stderr
        cen1x_1err[jj,ii]= out.params['cen_g1'].stderr
        wid1x_1err[jj,ii] = out.params['wid_g1'].stderr
        out_1gx = [amplitude, mean, stddev]
        bic_1gx = out.bic
        # chi1x = out.redchi
        
        # Pass the best fit parameters as initial parameters for the next pixel to fit
        fit_params1gx.add('amp_g1', value=out_1gx[0], min = 0.0025, max= 0.1)
        fit_params1g.add('cen_g1', value=out_1g[1], min = velmin, max= velmax)
        fit_params1gx.add('wid_g1', value=out_1gx[2], min = 10, max = 300)
        
        if printplot == True:
            # # res1[:,jj,ii] = residual(out.params, x, p, data, error/3) 
            plt.figure(figsize = (12,4))
            plt.plot(x, spec_tmp, label='data')
            plt.plot(x, data*0,':',color = 'black')
            plt.plot(x, fit1x, label='best fit')
            plt.xlabel('velocity [km/s]')
            plt.ylabel('flux [mJy]')
            plt.title('1Gaussian_model_M1_pixel' + str(ii) + '-' + str(jj) )
            plt.legend()
            plt.show()
        if printfit == True:
            print('##')
            print('1gaussian_fit pixel:' + str(ii) + '-' + str(jj))
            report_fit(out)
        
        ##FIT CON 2 GAUSSIANE with initial velocity taken from the "brute" moment 1
        p = 2
        n = 2
        fit_params2gx.add('cen_g1', value=M1[jj,ii], min = velmin, max= velmax)
        out = minimize(residual, fit_params2gx, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/2.5})
        fit2x = residual(out.params, x, p)
        # res2[:,jj,ii] = residual(out.params, x, p, data, error/3) 
        mod2x[:,jj,ii] = fit2x
        parvals = out.params.valuesdict()
        amplitude_2 = parvals['amp_g' + str(n)]
        stddev_2 = parvals['wid_g' + str(n)]
        mean_2 = parvals['cen_g' + str(n)]
        amplitude_1 = parvals['amp_g' + str(n-1)]
        stddev_1 = parvals['wid_g' + str(n-1)]
        mean_1 = parvals['cen_g' + str(n-1)]
        amp2x_1err[jj,ii] = out.params['amp_g1'].stderr
        cen2x_1err[jj,ii]= out.params['cen_g1'].stderr
        wid2x_1err[jj,ii] = out.params['wid_g1'].stderr
        amp2x_2err[jj,ii] = out.params['amp_g2'].stderr
        cen2x_2err[jj,ii]= out.params['cen_g2'].stderr
        wid2x_2err[jj,ii] = out.params['wid_g2'].stderr
        out_2gx = [amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2]
        bic_2gx = out.bic
        # chi2x = out.redchi
        
        # Pass the best fit parameters as initial parameters for the next pixel to fit
        fit_params2gx.add('amp_g1', value=out_2gx[0], min = 0.0025, max= 0.1)
        # fit_params2g.add('cen_g1', value=out_2g[1], min = velmin, max= velmax)
        fit_params2gx.add('wid_g1', value=out_2gx[2], min = 1, max = 200)
        fit_params2gx.add('amp_g2' , value=out_2gx[3], min= 0.0025, max= 0.1)
        fit_params2gx.add(name=('cen_g2'), expr='peak_split+cen_g1')
        fit_params2gx.add('wid_g2', value=out_2gx[5], min =1, max= 200)
        
        if printplot == True:
            plt.figure(figsize = (12,4))
            plt.plot(x, spec_tmp, label='data')
            plt.plot(x, data*0,':',color = 'black')
            plt.plot(x, fit2x, label='best fit')
            plt.xlabel('velocity [km/s]')
            plt.ylabel('flux [mJy]')
            plt.title('2Gaussian_model_M1_pixel' + str(ii) + '-' + str(jj) )
            plt.legend()
            plt.show()
        if printfit == True:
            print('##')
            print(str(n) + 'gaussianfit' + 'pixel:' + str(ii) + '-' + str(jj))
            report_fit(out)
        
        
        ##USE THE BIC FOR SELECTING THE BEST FIT
        if jj in range1 and ii in range2:
            # if bic_1g < bic_2g and bic_1g < bic_3g  and bic_2g - bic_1g > 2.3 and bic_3g - bic_1g > 2.3:
            bic_min = np.min([bic_1g, bic_2g, bic_1gx, bic_2gx])
            if bic_1g == bic_min:    
                flux_map[jj,ii] = np.nansum(fit1) * dv
                vel_map[jj,ii] = np.nansum((fit1*velocity)) * dv / flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit1*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
                vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                mod[:,jj,ii] = mod1[:,jj,ii]
                amperr[jj,ii] = amp1_1err[jj,ii]
                velerr[jj,ii] = cen1_1err[jj,ii]
                disperr[jj,ii] = wid1_1err[jj,ii]
            elif bic_2g == bic_min: 
                flux_map[jj,ii] = np.nansum(fit2) * dv
                vel_map[jj,ii] = np.nansum((fit2*velocity)) * dv/flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit2*(velocity-vel_map[jj,ii])**2)) * dv /flux_map[jj,ii]  
                vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                mod[:,jj,ii] = mod2[:,jj,ii]
                amperr[jj,ii] = np.sqrt(amp2_1err[jj,ii]**2 + amp2_2err[jj,ii]**2)
                velerr[jj,ii] = np.sqrt(cen2_1err[jj,ii]**2 + cen2_2err[jj,ii]**2)
                disperr[jj,ii] = np.sqrt(wid2_1err[jj,ii]**2 + wid2_2err[jj,ii]**2)
            elif bic_1gx == bic_min:
                flux_map[jj,ii] = np.nansum(fit1x) * dv 
                vel_map[jj,ii] = np.nansum((fit1x*velocity)) * dv / flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit1x*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
                vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                mod[:,jj,ii] = mod1x[:,jj,ii]
                amperr[jj,ii] = amp1x_1err[jj,ii]
                velerr[jj,ii] = cen1x_1err[jj,ii]
                disperr[jj,ii] = wid1x_1err[jj,ii]
            elif bic_2gx == bic_min:
                flux_map[jj,ii] = np.nansum(fit2x) * dv 
                vel_map[jj,ii] = np.nansum((fit2x*velocity)) * dv / flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit2x*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
                vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                mod[:,jj,ii] = mod2x[:,jj,ii]
                amperr[jj,ii] = np.sqrt(amp2x_1err[jj,ii]**2 + amp2x_2err[jj,ii]**2)
                velerr[jj,ii] = np.sqrt(cen2x_1err[jj,ii]**2 + cen2x_2err[jj,ii]**2)
                disperr[jj,ii] = np.sqrt(wid2x_1err[jj,ii]**2 + wid2x_2err[jj,ii]**2)
        
# flux_map[flux_map_tmp<5*error] = np.nan
# vel_map[flux_map_tmp < 5*error] = np.nan
# vdisp_map[flux_map_tmp < 5*error] = np.nan                 


# Plot            
plt.figure(figsize = (12,4))

plt.subplot(131)
plt.imshow(flux_map, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'Flux [mJ/y]')
plt.subplot(132)
plt.imshow(vel_map, origin = 'lower', vmin = -350, vmax = 350, cmap ='jet')
plt.colorbar(shrink = 0.7, label='Velocity [km/s]')
plt.subplot(133)
plt.imshow(vdisp_map, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'Velocity Dispersion [km/s]')

#----------------------------------------------------------------------------
# Saving the data
nn= 6
hdu = fits.PrimaryHDU(mod)
hdul = fits.HDUList([hdu])
hdul.writeto('model_' + str(nn) + '_2gx' + 'NGC6810' +'.fits', overwrite = True)


hdu = fits.PrimaryHDU(flux_map)
hdul = fits.HDUList([hdu])
hdul.writeto('flux_map_spiral_' + str(nn) +'_2gx' + 'NGC5643' + '.fits', overwrite = True)
hdu = fits.PrimaryHDU(vel_map)
hdul = fits.HDUList([hdu])
hdul.writeto('vel_map_spiral_'+ str(nn) +'_2gx' + 'NGC5643' + '.fits', overwrite = True)
hdu = fits.PrimaryHDU(vdisp_map)
hdul = fits.HDUList([hdu])
hdul.writeto('vdisp_map_spiral_'+ str(nn) +'_2g'  + 'NGC5643' + '.fits', overwrite = True)
hdu = fits.PrimaryHDU(amperr)
hdul = fits.HDUList([hdu])
hdul.writeto('flux_err_map_' + str(nn) +'_2gx' + 'NGC5643' + '.fits', overwrite = True)
hdu = fits.PrimaryHDU(velerr)
hdul = fits.HDUList([hdu])
hdul.writeto('vel_err_map_'+ str(nn) +'_2gx' + 'NGC5643' + '.fits', overwrite = True)
hdu = fits.PrimaryHDU(disperr)
hdul = fits.HDUList([hdu])
hdul.writeto('vdisp_err_map_'+ str(nn) +'_2g'  + 'NGC5643' + '.fits', overwrite = True)

#----------------------------------------------------------------------------







            # if bic_1g < bic_2g  and bic_2g - bic_1g <2.3:    
            #     flux_map[jj,ii] = np.nansum(fit1) * dv# out_1g[0] * np.sqrt(2 *np.pi) * out_1g[2]
            #     vel_map[jj,ii] = np.nansum((fit1*velocity)) * dv / flux_map[jj,ii]
            #     vdisp_map[jj,ii] = np.nansum((fit1*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
            #     vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
            #     # res[:,jj,ii] = res1[:,jj,ii]
            #     mod[:,jj,ii] = mod1[:,jj,ii]
            #     # print(flux_map[jj,ii], vel_map[jj,ii,], vdisp_map[jj,ii], '1g')
            #     # print('pixel:' + str(ii) + '-' + str(jj))
            #     # print(bic_1g, bic_2g, bic_3g, 'bestfit= 1G')
            #     # print(chi1,chi2,chi3)
            # # elif bic_2g < bic_1g and bic_2g < bic_3g and bic_1g - bic_2g > 2.3 and bic_3g - bic_2g > 2.3: #and bic_2g < bic_3g: # and bic_1g - bic_2g > 2.3
            # elif bic_2g < bic_1g and bic_1g - bic_2g > 2.3: #and bic_2g < bic_3g: # and bic_1g - bic_2g > 2.3  
            #     flux_map[jj,ii] = np.nansum(fit2) * dv# (out_2g[0] * np.sqrt(2 *np.pi) * out_1g[2] + out_2g[3]* np.sqrt(2 *np.pi) * out_1g[5])/2
            #     vel_map[jj,ii] = np.nansum((fit2*velocity)) * dv/flux_map[jj,ii]
            #     vdisp_map[jj,ii] = np.nansum((fit2*(velocity-vel_map[jj,ii])**2)) * dv /flux_map[jj,ii]  
            #     vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
            #     # res[:,jj,ii] = res2[:,jj,ii]
            #     mod[:,jj,ii] = mod2[:,jj,ii]
            #     # print(flux_map[jj,ii], vel_map[jj,ii,], vdisp_map[jj,ii], '2g')
            #     # print('pixel:' + str(ii) + '-' + str(jj))
            #     # print(bic_1g, bic_2g, bic_3g, 'bestfit= 2G')
            #     # print(chi1,chi2,chi3)
            # # elif bic_3g < bic_1g and bic_3g < bic_2g and bic_1g - bic_3g > 2.3 and bic_2g - bic_3g > 2.3:
            # #     flux_map[jj,ii] = np.nansum(fit3) * dv#(out_3g[0] * np.sqrt(2 *np.pi) * out_1g[2] + out_3g[3] *  np.sqrt(2 *np.pi) * out_1g[5] + out_3g[6] *  np.sqrt(2 *np.pi) * out_1g[8])/3
            # #     vel_map[jj,ii] = np.nansum((fit3*velocity)) * dv / flux_map[jj,ii]
            # #     vdisp_map[jj,ii] = np.nansum((fit3*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
            # #     vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
            # #     # res[:,jj,ii] = res3[:,jj,ii]
            # #     mod[:,jj,ii] = mod3[:,jj,ii]
            # #     # print(flux_map[jj,ii], vel_map[jj,ii,], vdisp_map[jj,ii], '3g')
            # #     # print('pixel:' + str(ii) + '-' + str(jj))
            # #     # print(bic_1g, bic_2g, bic_3g, 'bestfit= 3G')
            # #     # print(chi1,chi2,chi3)
            # # elif bic_2g < bic_1g and bic_2g < bic_3g and bic_1g - bic_2g < 2.3:
            # elif bic_2g < bic_1g and bic_1g - bic_2g < 2.3:
            #     flux_map[jj,ii] = np.nansum(fit1) * dv # out_1g[0] * np.sqrt(2 *np.pi) * out_1g[2]
            #     vel_map[jj,ii] = np.nansum((fit1*velocity)) * dv / flux_map[jj,ii]
            #     vdisp_map[jj,ii] = np.nansum((fit1*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
            #     vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
            #     # res[:,jj,ii] = res1[:,jj,ii]
            #     mod[:,jj,ii] = mod1[:,jj,ii]
            #     # print(flux_map[jj,ii], vel_map[jj,ii,], vdisp_map[jj,ii], '1g')
            #     # print('pixel:' + str(ii) + '-' + str(jj))
            #     # print(bic_1g, bic_2g, bic_3g, 'bestfit= 1G')
            #     # print(chi1,chi2,chi3)
            # # elif bic_3g < bic_1g and bic_3g < bic_2g and bic_2g - bic_3g < 2.3:
            # #     flux_map[jj,ii] = np.nansum(fit2) * dv# (out_2g[0] * np.sqrt(2 *np.pi) * out_1g[2] + out_2g[3]* np.sqrt(2 *np.pi) * out_1g[5])/2
            # #     vel_map[jj,ii] = np.nansum((fit2*velocity)) * dv / flux_map[jj,ii]
            # #     vdisp_map[jj,ii] = np.nansum((fit2*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]  
            # #     vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
            # #     # res[:,jj,ii] = res2[:,jj,ii]
            # #     mod[:,jj,ii] = mod2[:,jj,ii]
            # #     # print(flux_map[jj,ii], vel_map[jj,ii,], vdisp_map[jj,ii], '2g')
            # #     # print('pixel:' + str(ii) + '-' + str(jj))
            # #     # print(bic_1g, bic_2g, bic_3g, 'bestfit= 2G')
            # #     # print(chi1,chi2,chi3)


