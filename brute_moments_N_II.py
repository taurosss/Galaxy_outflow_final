#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:18:38 2023

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
    
    

    if p == 1:
        argu1 = (x - pars['cen_g1'])**2 / (2*(pars['wid_g1'])**2)
        model = pars['amp_g1'] * np.exp(-argu1) + pars['C_g1']
    if p == 2:
        argu1 = (x - pars['cen_g1'])**2 / (2*(pars['wid_g1'])**2)
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2'] * np.exp(-argu2)) + pars['C_g2']
    if p == 3:
        argu1 = (x - pars['cen_g1'])**2 / (2*(pars['wid_g1'])**2)
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3'] * np.exp(-argu3)) + pars['C_g3']
        model1 = pars['C_g3'] + pars['amp_g1'] * np.exp(-argu1)
        model1z = pars['amp_g1'] * np.exp(-argu1)
        model2z = pars['amp_g3'] * np.exp(-argu3)
    if p == 4:
        argu1 = (x - pars['cen_g1'])**2 / (2*(pars['wid_g1'])**2)
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        argu4 = (x - pars['cen_g4'])**2 / (2*(pars['wid_g4'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3']*np.exp(-argu3) + pars['amp_g4']*np.exp(-argu4)) + pars['C_g4']
    if p == 5:
        argu1 = (x - pars['cen_g1'])**2 / (2*(pars['wid_g1'])**2)
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        argu4 = (x - pars['cen_g4'])**2 / (2*(pars['wid_g4'])**2)
        argu5 = (x - pars['cen_g5'])**2 / (2*(pars['wid_g5'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3']*np.exp(-argu3) + pars['amp_g4']*np.exp(-argu4) + pars['amp_g5']*np.exp(-argu5)) + pars['C_g5']
    
    if data is None:
            return model, model1, model1z, model2z
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
filefits_data = 'NGC6810_Data.fits'
datacube = fits.open(path+'/file/Muse/'+filefits_data)[0]
model_cube = fits.open(path + '/2modalpha.fits')[0]
Nz,Ny,Nx = datacube.shape
print (Nz, Ny, Nx)


# define the z-axis which corresponds to frequency
naxis3 = datacube.header['NAXIS3']
crpix3 = datacube.header['CRPIX3']  # reference pixel
crval3 = datacube.header['CRVAL3']  # angstrom (Starting wavelenght)
cdelt3 = datacube.header['CD3_3']   # wavelenght increment

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
print(frequency_mean)


# z = v/c = (nu_emit - nu_obs)/nu_obs 
velocity_unit = ((frequency_mean- (frequency))/(frequency))*K.c.to('km/s')
velocity_beta = (((612.604172*u.THz - (frequency_beta))/(frequency_beta))*K.c.to('km/s')).value
velocity_O = (((594.850769*u.THz - (frequency_O))/(frequency_O))*K.c.to('km/s')).value
velocity_alpha = ((( 453.778836*u.THz - (frequency_alpha))/(frequency_alpha))*K.c.to('km/s')).value
velocity_N = ((( 452.3509988*u.THz - (frequency_alpha))/(frequency_alpha))*K.c.to('km/s')).value
print(velocity_unit[:10])
velocity = velocity_unit.value
print(velocity[:10])
dv = np.abs(velocity[1]-velocity[0])
dlambda = np.abs(wavelenght[1]-wavelenght[0])

#----------------------------------------------------------------------------

# TOTAL SPECTRUM
# location of the target
x0,y0 = 155, 150
# size of the square aperture 
dl = 80
# extract the spectrum
spectrum = np.nansum(datacube.data[:,y0-dl:y0+dl,x0-dl:x0+dl],axis = (1,2))

# 0plot: Wavelenght - Spectrum
plt.figure(figsize = (12,4))
plt.plot(wavelenght, spectrum, label = 'data')
plt.plot(wavelenght,wavelenght*0,':',color = 'black')
plt.xlabel('wavelenght [A°]')
plt.ylabel('flux ')
plt.title('Total Spectrum')
plt.legend()
plt.show()

# 1plot: frequency - spectrum
plt.figure(figsize = (12,4))
plt.plot(frequency, spectrum, label = 'data')
plt.plot(frequency,frequency*0,':',color = 'black')
plt.xlabel('frequency [THz]')
plt.ylabel('flux ')
plt.title('Total Spectrum')
plt.legend()
plt.show()


#----------------------------------------------------------------------------

## RMS DETERMINATION WITH THE POWER RESPONSE 

# Choosing an empty region
x0, y0 = 280, 30
dl = 15
noise = datacube.data[:,y0-dl:y0+dl,x0-dl:x0+dl]
error_beta = np.std(noise[8:98,:,:])
error_O = np.std(noise[ 110:210,:,:])
error_alpha = np.std(noise[1400:1500, :, :])
error_tot = np.std(noise[:, :, :])

print("rms  = {:2f} mJy".format(error_alpha))
print("####################")

#----------------------------------------------------------------------------

# Fit and plot of the total spectrum
# First Line: H_beta 4862
# Doublet O_III 4960 - 5008
# Doublet N_II 6549-6585 + H_alpha 6564
x = wavelenght
data = spectrum
p = 3

fit_params = Parameters()
fit_params.add('amp_g1', value=2e6,)
fit_params.add('cen_g1', value=4862.68)
fit_params.add('wid_g1', value=100)
fit_params.add('C_g1', value=1e5,)
fit_params.add('amp_g2', value=1e6,)
fit_params.add('cen_g2', value=5000)
fit_params.add('wid_g2', value=200)
fit_params.add('C_g2', value=1e5,)
fit_params.add('amp_g3', value=8e6,)
fit_params.add('cen_g3', value=6560)
fit_params.add('wid_g3', value=200)
fit_params.add('C_g3', value=1e5,)


out = minimize(residual, fit_params, args=(x,p,), kws={'data': data})
fit, fit1, fit1z, fit2z = residual(out.params, x, p)
print('##')
print('Total_Spectrum_fit')
report_fit(out)
parvals = out.params.valuesdict()
stddev_t = parvals['wid_g1']
print("FWHM  = {:2f} km/s".format(2.355*stddev_t))


bic_1g = out.bic            #Bayesian Crit Info for the fit with 1 Gaussian

plt.figure(figsize = (12,4))
plt.plot(x, data, label='data')
plt.plot(x, data*0,':',color = 'black')
plt.plot(x, fit, label='best fit')
plt.xlabel('Wavelenght [A°]')
plt.ylabel('flux [Jy]')
plt.title('Total Spectrum fit with 3 gaussians')
plt.legend()
plt.show()

del(x, fit, fit1, fit1z, fit2z)
#----------------------------------------------------------------------------

# Making 3 different subcubes for the 3 part of the fit (H_beta, O_III doublet, H_alpha e NII doublet)

#H_beta

O_III = datacube.data[8:98, :, :]
w_beta = wavelenght[8:98]
O_III = datacube.data[110:210, :, :]
w_O = wavelenght[110:210]
H_alpha = datacube.data[1400:1500, :, :]
w_alpha = wavelenght[1400:1500] #A°

#----------------------------------------------------------------------------

# Generating moments map without fitting (BRUTE MOMENTS MAP)
# H_beta

# #datacube.data = np.where(datacube.data<0, 0*datacube.data, datacube.data)
# mask_cube = np.where(H_alpha > 4.5*error_alpha, H_alpha, np.nan)
# M0 = np.nansum(mask_cube, axis = (0)) * dlambda
# M1 = np.nansum(mask_cube[:,:,:]*velocity_alpha[:,np.newaxis,np.newaxis], axis=0) *dlambda / M0
# # avoid division by 0 or neg values in sqrt
# # thr = np.nanpercentile(M0[np.where(M0>0)],0.01)
# thr = 4.5*error_alpha
# M0[np.where(M0<thr)]=np.nan
# M2 = np.sqrt(np.nansum(np.power(H_alpha[:, :, :] * (velocity_alpha[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :]),2), axis=0) *dlambda / M0 )     


# plt.figure(figsize = (12,4))

# plt.subplot(131)
# plt.imshow(M0, origin = 'lower', cmap = 'jet')
# plt.colorbar(shrink = 0.7)
# plt.title('Flux')
# plt.subplot(132)
# plt.imshow(M1, origin = 'lower',vmin= -1000, vmax= 1000, cmap ='jet')
# plt.colorbar(shrink = 0.7)
# plt.title('Velocity')
# plt.subplot(133)
# plt.imshow(M2, origin = 'lower',  vmin = 0, vmax= 2000, cmap = 'jet')
# plt.colorbar(shrink = 0.7)
# plt.title('Velocity Dispersion')
# plt.suptitle('brute moment maps for H_alpha doublet')

# plt.show()

#----------------------------------------------------------------------------

# Generete model maps 

mod = np.zeros_like(H_alpha)
mod1 = np.zeros_like(H_alpha)
mod1x = np.zeros_like(H_alpha)
mod_alpha = np.zeros_like(H_alpha)
mod_alphax = np.zeros_like(H_alpha)
mod_alpha_t = np.zeros_like(H_alpha)

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
flux_map_tmp = np.nansum(H_alpha, axis = (0))
flux_map = np.full_like(flux_map_tmp, np.nan)
flux_map_N = np.full_like(flux_map_tmp, np.nan)
vel_map = np.full_like(flux_map, np.nan)
vdisp_map = np.full_like(flux_map, np.nan)
amp1_1err= np.full_like(flux_map, np.nan)
cen1_1err= np.full_like(amp1_1err, np.nan)
wid1_1err= np.full_like(amp1_1err, np.nan)
# cost1_1err = np.full_like(amp1_1err, np.nan)
amp1x_1err=np.full_like(amp1_1err, np.nan)
cen1x_1err=np.full_like(amp1_1err, np.nan)
wid1x_1err=np.full_like(amp1_1err, np.nan)
# cost1x_1err = np.full_like(amp1_1err, np.nan)
amperr = np.full_like(amp1_1err, np.nan)
velerr = np.full_like(amp1_1err, np.nan)
disperr = np.full_like(amp1_1err, np.nan) 
# costerr = np.full_like(amp1_1err, np.nan)


# #----------------------------------------------------------------------------

#Initialize the initial parameters for the lmfit of the first pixel (the center)

fit_params3g = Parameters()
fit_params3g.add('amp_g1', value=4000, min =150)                   #H_alpha
fit_params3g.add('cen_g1', value= 6608.61,)
fit_params3g.add('wid_g1', value=1, min = 0.001, max = 500)
fit_params3g.add('C_g3', value = 30, min = 0 )
fit_params3g.add('amp_g2', value= 500, min =50)                    #N_II 6549
# fit_params3g.add('cen_g2', expr = 'cen_g1 + (6549.86 - 6564.61)/6564.61 * 299792.458 ')
fit_params3g.add('cen_g2', expr = 'cen_g1 * 6549.86 / 6564.61')
# fit_params3g.add('peak_split_1', value = -660, min = -720, max = -600)
# fit_params3g.add('cen_g2', expr = 'cen_g1 + peak_split_1')
fit_params3g.add('wid_g2', expr = 'wid_g1', min = 0.001, max = 500)
fit_params3g.add('amp_g3', expr = 'amp_g2 * 3', min =50,)             #N_II 6585
# fit_params3g.add('cen_g3', expr = 'cen_g1 + ( 6585.27 - 6564.61)/ 6564.61 * 299792.458')
fit_params3g.add('cen_g3', expr = 'cen_g1 * 6585.27 / 6564.61')
# fit_params3g.add('peak_split_2', value = 930, min = 870, max = 990)
# fit_params3g.add('cen_g3', expr = 'cen_g1 + peak_split_2')
fit_params3g.add('wid_g3', expr= 'wid_g1', min = 0.001, max = 500)


# #----------------------------------------------------------------------------

# # velmin, velmax = -300, 300
range1 = list(range(0,320))        # y range for the final maps  
range2 = list(range(0,318))       # x range for the final maps
printplot = True
printfit = True
    



# Substract the model cube of the fitted H_alpha to make the brute moments map of N_II 
N_total_cube = H_alpha - model_cube.data
w_N = w_alpha[50:]
v_N = velocity_N[50:]
N_cube = N_total_cube[50:, : , :]


# Generating moments map without fitting (BRUTE MOMENTS MAP)
# H_beta

#datacube.data = np.where(datacube.data<0, 0*datacube.data, datacube.data)
mask_cube = np.where(N_cube > 5*error_alpha, N_cube, np.nan)
M0 = np.nansum(mask_cube, axis = (0)) * dlambda
M1 = np.nansum(mask_cube[:,:,:]*v_N[:,np.newaxis,np.newaxis], axis=0) *dlambda / M0
# avoid division by 0 or neg values in sqrt
# thr = np.nanpercentile(M0[np.where(M0>0)],0.01)
thr = 5*error_alpha
M0[np.where(M0<thr)]=np.nan
# M2 = np.sqrt(np.nansum(np.power(mask_cube[:, :, :] * (v_N[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :]),2), axis=0) *dlambda / M0 )   
M2 = np.sqrt(np.nansum(mask_cube[:, :, :] * np.power((v_N[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :]),2), axis=0) *dlambda / M0 )    

# M2_2 = np.ndarray([318,320])
# M2 = np.ndarray([318,320])
# for jj in range(Ny):
#     for ii in range(Nx):
#         spec_tmp = N_cube[:,jj,ii]
#         M2_2[jj,ii] = (np.nansum( mask_cube[:,jj,ii] * (v_N[:] - M1[jj,ii])**2))*dlambda / M0
#         M2[jj,ii] = np.sqrt(M2_2)
        
        


plt.figure(figsize = (12,4))

plt.subplot(131)
plt.imshow(M0, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7)
plt.title('Flux')
plt.subplot(132)
plt.imshow(M1, origin = 'lower',vmin= -500, vmax= 500, cmap ='jet')
plt.colorbar(shrink = 0.7)
plt.title('Velocity')
plt.subplot(133)
plt.imshow(M2, origin = 'lower',  vmin = 0, vmax= 400, cmap = 'jet')
plt.colorbar(shrink = 0.7)
plt.title('Velocity Dispersion')
plt.suptitle('brute moment maps for N_II 6585 A° line')

plt.show()
#----------------------------------------------------------------------------
# x = w_alpha
# p = 3
# # Spiral Fit
# for idxs in spiral_vect[0:5]:
#     prt = idxs.partition("-")                       # Take the coordinates of the pixel in the spiral fitting
#     ii = int(prt[2])    #x
#     jj = int(prt[0])    #y
#     spec_tmp = H_alpha[:,jj,ii]
#     spec_tmp = np.nan_to_num(spec_tmp)
#     if flux_map_tmp[jj,ii] > 10 * error_alpha:    
#         # FIT WITH 2 GAUSSIAN
#         out = minimize(residual, fit_params3g, args=(x,p,), kws={'data': spec_tmp,'sigma': error_alpha * 10})
#         fit, fit1, fit1z, fit_N = residual(out.params, x, p)
#         mod1[:,jj,ii] = fit
#         mod_alpha[:, jj, ii] = fit1
#         parvals = out.params.valuesdict()
#         amplitude_1 = parvals['amp_g1']
#         stddev_1 = parvals['wid_g1']
#         mean_1 = parvals['cen_g1']
#         cost = parvals['C_g3']
#         amplitude_2 = parvals['amp_g2']
#         stddev_2 = parvals['wid_g2']
#         mean_2 = parvals['cen_g2' ]
#         amplitude_3 = parvals['amp_g3']
#         stddev_3 = parvals['wid_g3']
#         mean_3 = parvals['cen_g3' ]

#         amp1_1err[jj,ii] = out.params['amp_g1'].stderr
#         cen1_1err[jj,ii]= out.params['cen_g1'].stderr
#         wid1_1err[jj,ii] = out.params['wid_g1'].stderr

#         out_1g = [amplitude_1, mean_1, stddev_1, cost, amplitude_2, mean_2, stddev_2, amplitude_3, mean_3, stddev_3]
#         bic_1g = out.bic
        
#         # Pass the best fit parameters as initial parameters for the next pixel to fit
#         fit_params3g.add('amp_g1', value=out_1g[0], min =100)                   #H_alpha
#         fit_params3g.add('cen_g1', value=out_1g[1])
#         fit_params3g.add('wid_g1', value=out_1g[2], min = 0.001, max = 500)
#         fit_params3g.add('C_g3', value = out_1g[3], min = 0 , max = 300)
#         fit_params3g.add('amp_g2', value= out_1g[4], min =45)                    #N_II 6549
#         # fit_params3g.add('peak_split_1', value = split1)
#         # fit_params3g.add('cen_g2', expr = 'cen_g1 + peak_split_1')
#         # fit_params3g.add('cen_g2', expr = 'cen_g1 + (6549.86 - 6564.61)/6564.61 * 299792.458 ')
#         fit_params3g.add('cen_g2', expr = 'cen_g1 * 6549.86 / 6564.61')
#         fit_params3g.add('wid_g2', expr = 'wid_g1', min = 0.001, max = 500)
#         fit_params3g.add('amp_g3', expr = 'amp_g2 * 3', min =50)             #N_II 6585
#         # fit_params3g.add('peak_split2', value = split2)
#         # fit_params3g.add('cen_g3', expr = 'cen_g1  + peak_split_2')
#         # fit_params3g.add('cen_g3', expr = 'cen_g1 + ( 6585.27 - 6564.61)/ 6564.61 * 299792.458')
#         fit_params3g.add('cen_g3', expr = 'cen_g1 * 6585.27 / 6564.61')
#         fit_params3g.add('wid_g3', expr= 'wid_g1', min = 0.001, max = 500)
        
        
#         if printplot == True:
#             plt.figure(figsize = (12,4))
#             plt.plot(x, spec_tmp, label='data')
#             plt.plot(x, x*0,':',color = 'black')
#             plt.plot(x, fit, label='best fit')
#             plt.plot(x, fit1, label = 'alpha line')
#             plt.xlabel('velocity [km/s]')
#             plt.ylabel('flux [Jy]')
#             plt.title('3gaussian_model_pixel' + str(ii) + '-' + str(jj) )
#             plt.legend()
#             plt.show()
#         if printfit == True:
#             print('##')
#             print('3gaussian_fit pixel:' + str(ii) + '-' + str(jj))
#             report_fit(out)
        
        
        
#         # ##USE THE BIC FOR SELECTING THE BEST FIT
#         # if jj in range1 and ii in range2:
#         #     # if bic_1g < bic_2g and bic_1g < bic_3g  and bic_2g - bic_1g > 2.3 and bic_3g - bic_1g > 2.3:
#         #     bic_min = np.min([bic_1g, bic_1gx])
#         #     if bic_1g == bic_min:    
#         #         flux_map[jj,ii] = np.nansum(fit1z) * dlambda
#         #         vel_map[jj,ii] = np.nansum((fit1z*velocity_alpha)) * dlambda / flux_map[jj,ii]
#         #         vdisp_map[jj,ii] = np.nansum((fit1z*(velocity_alpha-vel_map[jj,ii])**2)) * dlambda / flux_map[jj,ii]
#         #         vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
#         #         mod[:,jj,ii] = mod1[:,jj,ii]
#         #         mod_alpha_t[:,jj,ii] = mod_alpha[:,jj,ii]
#         #         amperr[jj,ii] = amp1_1err[jj,ii]
#         #         velerr[jj,ii] = cen1_1err[jj,ii]
#         #         disperr[jj,ii] = wid1_1err[jj,ii]
#         #         flux_map_N[jj,ii] = np.nansum(fit_N) * dlambda
#         #         # costerr[jj,ii] = cost1_1err[jj,ii]
#         #     elif bic_1gx == bic_min:
#         #         flux_map[jj,ii] = np.nansum(fit1xz) * dlambda 
#         #         vel_map[jj,ii] = np.nansum((fit1xz*velocity_alpha)) * dlambda / flux_map[jj,ii]
#         #         vdisp_map[jj,ii] = np.nansum((fit1xz*(velocity_alpha-vel_map[jj,ii])**2)) * dlambda / flux_map[jj,ii]
#         #         vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
#         #         mod[:,jj,ii] = mod1x[:,jj,ii]
#         #         mod_alpha_t[:,jj,ii] = mod_alphax[:,jj,ii]
#         #         amperr[jj,ii] = amp1x_1err[jj,ii]
#         #         velerr[jj,ii] = cen1x_1err[jj,ii]
#         #         disperr[jj,ii] = wid1x_1err[jj,ii]
#         #         flux_map_N[jj,ii] = np.nansum(fitx_N) * dlambda
#         #         # costerr[jj,ii] = cost1x_1err[jj,ii]
        
# # flux_map[flux_map_tmp<3.5*error_alpha] = np.nan
# # vel_map[flux_map_tmp < 3.5*error_alpha] = np.nan
# # vdisp_map[flux_map_tmp < 3.5*error_alpha] = np.nan                 


# # Plot            
# plt.figure(figsize = (20,6))

# plt.subplot(131)
# plt.imshow(flux_map, origin = 'lower', cmap = 'jet')
# plt.colorbar(shrink = 0.7, label = 'Flux [Jy]')
# plt.title('Flux')
# plt.subplot(132)
# plt.imshow(vel_map, origin = 'lower', vmin = -500, vmax = 500, cmap ='jet')
# plt.colorbar(shrink = 0.7, label='Velocity [km/s]')
# plt.title('Velocity')
# plt.subplot(133)
# plt.imshow(vdisp_map, origin = 'lower',  vmin = 0, vmax=600, cmap = 'jet')
# plt.colorbar(shrink = 0.7, label = 'Velocity Dispersion [km/s]')
# plt.title('Velocity Dispersion')
# plt.suptitle('Moments Maps')

# plt.figure(figsize = (12,4))
# plt.imshow(flux_map_N, origin = 'lower', cmap = 'jet')
# plt.colorbar(label = 'Flux [Jy]')
# plt.title('Flux N_II 6585')
# #----------------------------------------------------------------------------
# Saving the data
 
# hdu = fits.PrimaryHDU(mod)
# hdul = fits.HDUList([hdu])
# hdul.writeto('1mod.fits', overwrite = True)
# hdu = fits.PrimaryHDU(mod_alpha_t)
# hdul = fits.HDUList([hdu])
# hdul.writeto('2modalpha.fits', overwrite = True)
# hdu = fits.PrimaryHDU(flux_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('3flux_map_H.fits', overwrite = True)
# hdu = fits.PrimaryHDU(vel_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('4vel_H.fits', overwrite = True)
# hdu = fits.PrimaryHDU(vdisp_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('5disp_H.fits', overwrite = True)

# hdu = fits.PrimaryHDU(amperr)
# hdul = fits.HDUList([hdu])
# hdul.writeto('6amperr_H.fits', overwrite = True)
# hdu = fits.PrimaryHDU(velerr)
# hdul = fits.HDUList([hdu])
# hdul.writeto('7velerr_H.fits', overwrite = True)
# hdu = fits.PrimaryHDU(disperr)
# hdul = fits.HDUList([hdu])
# hdul.writeto('8disperr_H.fits', overwrite = True)
# hdu = fits.PrimaryHDU(flux_map_N)
# hdul = fits.HDUList([hdu])
# hdul.writeto('9flux_map_N.fits', overwrite = True)

# #----------------------------------------------------------------------------


# tmod = fits.open(path + '/1.fits')[0].data
# tmod_alpha_t = fits.open(path + '/2.fits')[0].data
# tflux_map = fits.open(path + '/3.fits')[0].data
# tvel_map = fits.open(path + '/4.fits')[0].data
# tvdisp_map =fits.open(path + '/5.fits')[0].data
# tamperr =fits.open(path + '/6.fits')[0].data
# tvelerr = fits.open(path + '/7.fits')[0].data
# tdisperr = fits.open(path + '/8.fits')[0].data
# tcosterr = fits.open(path + '/9.fits')[0].data

# mask = np.isnan(flux_map)
# # mask1 = np.where(mod==0)

# # mod[np.where(mod==0)] = tmod
# # mod_alpha_t[np.where(mod==0)] = tmod_alpha_t
# flux_map[mask] = tflux_map[mask]
# vel_map[mask] = tvel_map[mask]
# vdisp_map[mask] = tvdisp_map[mask]
# amperr[mask] = tamperr[mask]
# velerr[mask] = tvelerr[mask]
# disperr[mask] = tdisperr[mask]
# costerr[mask] =tcosterr[mask]





