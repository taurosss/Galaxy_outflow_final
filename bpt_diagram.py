#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:58:08 2023

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
import matplotlib.colors as mcolors



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



# # 2plot: velocity - spectrum
# plt.figure(figsize = (12,4))
# plt.plot(velocity, spectrum, label = 'data')
# plt.plot(velocity,frequency*0,':',color = 'black')
# plt.xlabel('velocity [km/s]')
# plt.ylabel('flux ')
# plt.title('Totale Spectrum (function of velocity)')
# plt.legend()
# plt.show()

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
w_alpha = wavelenght[1400:1500]


#----------------------------------------------------------------------------

mod = fits.open(path + '/1mod.fits')[0].data
mod_alpha_t = fits.open(path + '/2modalpha.fits')[0].data
flux_H_alpha = fits.open(path + '/flux_map_H_definitive.fits')[0].data
vel_map = fits.open(path + '/4vel_H.fits')[0].data
vdisp_map =fits.open(path + '/5disp_H.fits')[0].data
amperr =fits.open(path + '/6amperr_H.fits')[0].data
velerr = fits.open(path + '/7velerr_H.fits')[0].data
disperr = fits.open(path + '/8disperr_H.fits')[0].data 
flux_NII = fits.open(path + '/9flux_map_N.fits')[0].data
flux_OIII = fits.open(path + '/moments_map/NGC6810_MUSE/O_III/flux_map_O_5008_spiral_3_1gxNGC6810_MUSE_OIII.fits')[0].data
flux_H_beta = fits.open(path + '/moments_map/NGC6810_MUSE/H_beta/flux_map_spiral_4_1gxNGC6810_MUSE_H_beta_clean.fits')[0].data

flux_H_alpha *= 1e-20
flux_H_beta *= 1e-20
flux_NII *= 1e-20
flux_OIII *= 1e-20
# Plot            
plt.figure(figsize = (14,9))

plt.subplot(221)
plt.imshow(flux_H_alpha, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'Flux [Jy]')
plt.title('H_alpha')
plt.subplot(222)
plt.imshow(flux_NII, origin = 'lower', cmap ='jet')
plt.colorbar(shrink = 0.7, label='Flux [Jy]')
plt.title('N_II')
plt.subplot(223)
plt.imshow(flux_OIII, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'Flux [Jy]')
plt.title('O_III')
plt.subplot(224)
plt.imshow(flux_H_beta, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'Flux [Jy]')
plt.title('H_beta')
plt.suptitle('Flux Maps')
plt.show()


mask1 = ~np.isnan(flux_H_alpha) 
mask2 = ~np.isnan(flux_NII)
mask3 = ~np.isnan(flux_OIII)
mask4 = ~np.isnan(flux_H_beta)
mask = mask1 * mask2 * mask3 * mask4
map_x = np.full_like(flux_H_alpha, np.nan)
map_y = np.full_like(flux_H_alpha, np.nan)
map_x = np.log10(flux_NII/flux_H_alpha)
map_y = np.log10(flux_OIII /flux_H_beta)
x = np.log10(flux_H_alpha[mask] /flux_NII[mask])
y = np.log10(flux_OIII[mask] /flux_H_beta[mask])
log_nii_ha = np.log10(flux_NII)
log_OIII = np.log10(flux_OIII)
log_hb = np.log10(flux_H_beta)
log_ha = np.log10(flux_H_alpha)







# Below are listed the demarcations summarized by Kewley et al 2006 for each diagram:
# 1) BPT-NII:
#     log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.05) + 1.3     (Kauffmann+03 line)
#     log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.47) + 1.19    (Kewley+01 line)
# 2) BPT-SII:
#     log([OIII]/Hb) = 0.72 / (log([SII]/Ha) - 0.32) + 1.30    (main AGN line)
#     log([OIII]/Hb) = 1.89 log([SII]/Ha) + 0.76   (LINER/Sy2 line)
# 3) BPT-OI:
#     log([OIII]/Hb) = 0.73 / (log([OI]/Ha) + 0.59) + 1.33    (main AGN line)
#     log([OIII]/Hb) = 1.18 log([OI]/Ha) + 1.30  (LINER/Sy2 line)

log_nii_ha = np.log10(flux_NII)
log_OIII = np.log10(flux_OIII)
log_hb = np.log10(flux_H_beta)
log_ha = np.log10(flux_H_alpha)

def bpt_nii(log_nii_ha, relation = 'both'):
    log_nii_ha_k03 = np.copy(log_nii_ha)
    log_nii_ha_k03[log_nii_ha_k03>= 0.05] = 0.04
    log_nii_ha_k01 = np.copy(log_nii_ha)
    log_nii_ha_k01[log_nii_ha_k01>= 0.47] = 0.46
    kauffmann03 =  0.61 / (log_nii_ha_k03 - 0.05) + 1.3
    kewley01 =  0.61 / (log_nii_ha_k01  - 0.47) + 1.19
    #kauffman < kewley
    log_oiii_ha = np.asarray([kauffmann03,kewley01])
    return log_oiii_ha
def pbt(x_ratio, y_ratio, diagram = 'nii', quiet = True):
    '''
    Parameters
    ----------
    x_ratio : array
        log10([OIII]/Hb)
    y_ratio : array
        log10([NII]/Ha) o
    diagram : strinf, optional
        select the PBT diagram to use for the selection.
        The default is 'nii’.
    quiet : TYPE, optional
        DESCRIPTION. The default is True.
    Returns
    -------
    bpt_type : array
        return the BPT classification
        1: SF
        2: mixed
        3: AGN
    '''
    if diagram == 'nii':
        y1_ratio, y2_ratio = bpt_nii(x_ratio, relation = 'both')
        sel_kauffmann03 = y_ratio>y1_ratio
        sel_kewley01 = y_ratio>y2_ratio
        bpt_type = sel_kauffmann03.astype(int)+sel_kewley01.astype(int)
    return bpt_type
def PBTdiagrams(oiiihb,niiha, siiha,oiha, color = 'blue',
                ax= None):
#    ax.subplot(131)
#    p1 = plt.scatter(niihaN,oiiihbN, color = 'orange’, label = 'narrow’)
    ax[0].scatter(niiha, oiiihb ,c = color)#,color = 'blue’, label = 'broad’)
    x = np.linspace(-2,1,100)
    y1 = 0.61/(x-0.47)+1.19
    y1[x>=0.47]=-12
    y2 = 0.61/(x-0.05)+1.3
    y2[x>=0.05]=-12
    ax[0].plot(x,y1, color='black')
    ax[0].plot(x,y2, color='black', linestyle='--')
    ax[0].set_xlabel(r'Log([NII]6584/H$\alpha$)')
    ax[0].set_ylabel(r'Log([OIII]5007/H$\beta$)')
    ax[0].set_xlim(-1.5,0.8)
    ax[0].set_ylim(-0.4,1.2)
#    ax.subplot(132)
    ax[1].scatter(siiha, oiiihb,  c = color)#,color = 'blue’)
    ax[1].set_xlim(-1.5,0.6)
    ax[1].set_ylim(-0.4,1.2)
    x = np.linspace(-2,1,100)
    y1 = 0.72/(x-0.32)+1.30
    y1[x>=0.32]=-12
    y2 = 1.89*x+0.76
    y2[y2<=y1]=y1[y2<=y1]
    ax[1].set_xlabel(r'Log([SII]6717+6731/H$\alpha$)')
    ax[1].set_ylabel(r'Log([OIII]5007/H$\beta$)')
    ax[1].plot(x,y1, color='black')
    ax[1].plot(x,y2, color='black', linestyle='--')
#    ax.subplot(133)
#    plt.scatter(oihaN,oiiihbN, color = 'orange')
    ax[2].scatter(oiha, oiiihb, c = color)#,color = 'blue')   #mettere una lista di 1 al post do oiha
    x = np.linspace(-3,1,100)
    y1 = 0.73/(x+0.59)+1.33
    y1[x>=-0.59]=-12
    y2 = 1.18*x+1.30
    y2[y2<=y1]=y1[y2<=y1]
    ax[2].set_xlabel(r'Log([OI]6300/H$\alpha$)')
    ax[2].set_ylabel(r'Log([OIII]5007/H$\beta$)')
    ax[2].plot(x,y1, color='black')
    ax[2].plot(x,y2, color='black', linestyle='--')
    ax[2].set_xlim(-2.5,-0.0)
    ax[2].set_ylim(-0.4,1.2)

# log_nii_ha = np.arange(-2,1,0.1)
# y1,y2 = bpt_nii(log_nii_ha, relation = 'both')
# log_nii_ha_obs = np.asarray([0.5,-0.4,-1])
# log_oiii_hb_obs = np.asarray([1,0.4,0.4])
# print(pbt(log_nii_ha_obs,log_oiii_hb_obs))
# plt.plot(log_nii_ha,y1)
# plt.plot(log_nii_ha,y2)
# plt.scatter(log_nii_ha_obs,log_oiii_hb_obs)
# plt.ylim(-0.5,1.2)




log_nii_ha = np.arange(-2,1,0.1)
y1,y2 = bpt_nii(log_nii_ha, relation = 'both')
log_nii_ha_obs = np.asarray([0.5,-0.4,-1])
log_oiii_hb_obs = np.asarray([1,0.4,0.4])
print(pbt(log_nii_ha_obs,log_oiii_hb_obs))
plt.plot(log_nii_ha,y1)
plt.plot(log_nii_ha,y2)
plt.scatter(log_nii_ha_obs,log_oiii_hb_obs)
plt.ylim(-0.5,1.2)



mask1 = ~np.isnan(flux_H_alpha) 
mask2 = ~np.isnan(flux_NII)
mask3 = ~np.isnan(flux_OIII)
mask4 = ~np.isnan(flux_H_beta)
mask = mask1 * mask2 * mask3 * mask4
x = np.log10(flux_NII[mask]/flux_H_alpha[mask] )
y = np.log10(flux_OIII[mask] /flux_H_beta[mask])

# log_nii_ha = np.log10(flux_NII)
log_OIII = np.log10(flux_OIII)
log_hb = np.log10(flux_H_beta)
log_ha = np.log10(flux_H_alpha)
bpttypes = pbt(x, y)
sfmask = np.zeros_like(bpttypes)
sfmask[bpttypes == 0] = True
mixedmask = np.zeros_like(bpttypes)
mixedmask [bpttypes == 1 ] = True
agnmask = np.zeros_like(bpttypes)
agnmask [bpttypes == 2 ] = True


plt.figure(figsize = (14,10))
plt.scatter(x, y, label='data', marker = '.')
plt.xlabel('$log_{10}$([$N_{II}$]/[$H_{alpha}$])')
plt.ylabel('$log_{10}$([$O_{III}$]/[$H_{beta}$])')
plt.title('BPT Diagram')
plt.plot(log_nii_ha,y1, 'o-', color='red')
plt.plot(log_nii_ha,y2,'o-', color='orange')
# plt.scatter(log_nii_ha_obs,log_oiii_hb_obs)
plt.ylim(-2,2)
plt.xlim(-2, 1)
plt.legend()
plt.show()

# plt.figure(figsize = (14,10))
# plt.scatter(x[sfmask], y[sfmask], label='SF', marker = '.', color= 'blue')
# plt.scatter(x[mixedmask], y[mixedmask], label='Mixed', marker = '.', color='green')
# plt.scatter(x[agnmask], y[agnmask], label='AGN', marker = '.', color='red')
# plt.xlabel('log10([H_alpha]/[N_II])')
# plt.ylabel('log10([O_III]/[H_beta])')
# plt.title('BPT Diagram')
# plt.plot(log_nii_ha,y1, 'o-', color='red')
# plt.plot(log_nii_ha,y2,'o-', color='orange')
# # plt.scatter(log_nii_ha_obs,log_oiii_hb_obs)
# plt.ylim(-2,2)
# plt.xlim(-2, 1)
# plt.legend()
# plt.show()


#prepare for masking arrays - 'conventional' arrays won't do it
y_values = np.ma.array(y)
#mask values below a certain threshold
y_sf = np.ma.masked_where(bpttypes < 0 , y_values)
y_mixed = np.ma.masked_where(bpttypes < 1 , y_values)
y_agn = np.ma.masked_where(bpttypes < 2 , y_values)

plt.figure(figsize = (14,10))
plt.scatter(x, y_sf, label='SF', marker = '.', color= 'blue')
plt.scatter(x, y_mixed, label='Mixed', marker = '.', color='green')
plt.scatter(x, y_agn, label='AGN', marker = '.', color='red')
plt.xlabel('$log_{10}$([$N_{II}$]/[$H_{alpha}$])')
plt.ylabel('$log_{10}$([$O_{III}$]/[$H_{beta}$])')
plt.title('BPT Diagram')
plt.plot(log_nii_ha,y1, 'o-', color='orange', label = 'kauffmann03')
plt.plot(log_nii_ha,y2,'s-', color='black', label = 'kewley01')
# plt.scatter(log_nii_ha_obs,log_oiii_hb_obs)
plt.ylim(-2,2)
plt.xlim(-2, 1)
plt.legend()
plt.show()


#Remapping the bpt diagram on the galaxy

mask_map = np.ma.array(map_y)
log_nii_ha_k03 = np.copy(map_x)
log_nii_ha_k03[log_nii_ha_k03>= 0.05] = 0.04
log_nii_ha_k01 = np.copy(map_x)
log_nii_ha_k01[log_nii_ha_k01>= 0.47] = 0.46
kauffmann03 = y1_ratio = 0.61 / (log_nii_ha_k03 - 0.05) + 1.3
kewley01 = y2_ratio = 0.61 / (log_nii_ha_k01  - 0.47) + 1.19
cond1 = map_y < y1_ratio
cond2 = (map_y > y1_ratio) * (map_y < y2_ratio)
cond3 = map_y > y2_ratio
map_sf = np.ma.masked_where(cond1  , mask_map)
map_mixed = np.ma.masked_where(cond2, mask_map)
map_agn = np.ma.masked_where(cond3, mask_map)

# map_sf = np.ma.masked_where((map_y < y1_ratio).any  , mask_map)
# map_mixed = np.ma.masked_where((map_y > y1_ratio).any and (map_y < y2_ratio).any, mask_map)
# map_agn = np.ma.masked_where((map_y > y2_ratio).any, mask_map)

map_sf = 1.0 * np.ones_like(map_y)
map_sf[cond1 == False] = np.nan
map_mixed = 1.0 * np.ones_like(map_y)
map_mixed[cond2 == False] = np.nan
map_agn = 1.0 * np.ones_like(map_y)
map_agn[np.where(cond3 == False)] = np.nan

#Plot

plt.figure(figsize = (14,9))

test = 1* cond3
plt.imshow(flux_H_alpha, origin='lower', cmap = 'gray', alpha = 0.1)
plt.imshow(map_sf, origin = 'lower', cmap = 'Blues',interpolation = 'none')
plt.colorbar(shrink = 0.7, label = 'SF')
plt.imshow(map_mixed, origin = 'lower', cmap ='Greens',interpolation = 'none')
plt.colorbar(shrink = 0.7, label='Mixed')
plt.imshow(map_agn, origin = 'lower', cmap = 'Reds', interpolation = 'none')
plt.colorbar(shrink = 0.7, label = 'AGN')
plt.title('BPT Diagram projected on the Galaxy')
plt.show()


# --------------------------------------------------
# # Saving the data
 
# hdu = fits.PrimaryHDU(mod)
# hdul = fits.HDUList([hdu])
# hdul.writeto('1mod.fits', overwrite = True)
# #----------------------------------------------------------------------------

