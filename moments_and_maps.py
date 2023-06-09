#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tauro
"""

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
from heapq import nlargest

path = os.path.dirname(os.path.abspath('__file__'))


filefits_data = 'NGC6810_crop.fits'
filefits_antenna = 'NGC6810_antenna.fits'
datacube = fits.open(path+'/file/'+filefits_data)[0]
datacube_antenna = fits.open(path+'/file/'+filefits_antenna)[0]
datacube.data = np.squeeze(datacube.data)
datacube_antenna.data = np.squeeze(datacube_antenna.data)
Nz,Ny,Nx = datacube.shape
print (Nz, Ny, Nx)


#define the z-axis which corresponds to frequency
naxis3 = datacube.header['NAXIS3']
crpix3 = datacube.header['CRPIX3']
crval3 = datacube.header['CRVAL3']
cdelt3 = datacube.header['CDELT3']

kk = 1+np.arange(naxis3)
            
frequency = crval3+cdelt3*(kk-crpix3) #Hz
frequency /= 1e9 #GHz

print(frequency[:10])


#define the z-axis in velocity units 
#average frequency
frequency_mean = np.mean(frequency)*u.GHz
print(frequency_mean)




#z = v/c = (nu_emit - nu_obs)/nu_obs 
velocity_unit = ((frequency_mean- (frequency*u.GHz))/(frequency*u.GHz))*K.c.to('km/s')
print(velocity_unit[:10])
velocity = velocity_unit.value
print(velocity[:10])
dv = velocity[1]-velocity[0]


#location of the target
x0,y0 = 250, 250
#size of the square aperture 
dl = 100
#extract the spectrum
#total spectrum
spectrum = np.nansum(datacube.data[:,y0-dl:y0+dl,x0-dl:x0+dl],axis = (1,2))
#1plot: frequency - spectrum
plt.plot(frequency, spectrum, label = 'data')
plt.plot(frequency,frequency*0,':',color = 'black')
plt.xlabel('frequency [GHz]')
plt.ylabel('flux ')
plt.title('Total Spectrum')
plt.legend()
plt.show()


#2plot: velocity - spectrum
plt.figure(figsize = (12,4))
plt.plot(velocity, spectrum, label = 'data')
plt.plot(velocity,frequency*0,':',color = 'black')
plt.xlabel('velocity [km/s]')
plt.ylabel('flux ')
plt.title('Totale Spectrum (function of velocity)')
plt.legend()
plt.show()




## RMS DETERMINATION WITH THE POWER RESPONSE 

#data/power response
noise_cube = datacube.data / datacube_antenna.data

#Choosing an empty region
x0, y0 = 294, 143
dl = 20
noise = noise_cube[:,y0-dl:y0+dl,x0-dl:x0+dl]
for ii in range(dl):
    for jj in range(dl):
            noise[0, jj, ii] = 0
#frequency_window = frequency[:,y0-dl:y0+dl,x0-dl:x0+dl]
noise_spectrum = np.nansum(noise,axis = (0))
#number of pixel selected
N = (2*dl)**2
for ii in range(2*dl):
    for jj in range(2*dl):
        rms = np.sum((noise_spectrum[ii,jj]**2))
        
rms = np.sqrt(rms/N)
error = rms
print("rms  = {:2f} mJy".format(rms))
print("####################")



## Multi-gaussians model

#define the "correct" paramters
p_true = Parameters()
p_true.add('amp_g1', value=0.1)
p_true.add('cen_g1', value=0.)
p_true.add('wid_g1', value=200.)
p_true.add('amp_g2', value=0.1)
p_true.add('cen_g2', value=0.)
p_true.add('wid_g2', value=0.1)
p_true.add('amp_g3', value=0.1)
p_true.add('cen_g3', value=0.)
p_true.add('wid_g3', value=200)
p_true.add('amp_g4', value=0.1)
p_true.add('cen_g4', value=0.)
p_true.add('wid_g4', value=200)
p_true.add('amp_g5', value=0.1)
p_true.add('cen_g5', value=0.)
p_true.add('wid_g5', value=200)




def residual(pars, x, p, data=None, sigma=None):
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

x = velocity
data = spectrum
p = 1

fit_params = Parameters()
fit_params.add('amp_g1', value=50,)
fit_params.add('cen_g1', value=0)
fit_params.add('wid_g1', value=100)

out = minimize(residual, fit_params, args=(x,p,), kws={'data': data,'sigma': error})
fit = residual(out.params, x, p)
print('##')
print('1gaussian_fit')
report_fit(out, modelpars=p_true)
parvals = out.params.valuesdict()
stddev_t = parvals['wid_g1']
print("FWHM  = {:2f} km/s".format(2.355*stddev_t))

#Bayesian Crit Info for the fit with 1 Gaussian
bic_1g = out.bic

plt.figure(figsize = (12,4))
plt.plot(x, data, label='data')
plt.plot(x, data*0,':',color = 'black')
plt.plot(x, fit, label='best fit')
plt.xlabel('velocity [km/s]')
plt.ylabel('flux [mJy]')
plt.title('1Gaussian_model')
plt.legend()
plt.show()

#generate moments map

flux_map_tmp = np.nansum(datacube.data, axis = (0))
flux_map = np.zeros_like(flux_map_tmp)
vel_map = np.zeros_like(flux_map)
vdisp_map = np.zeros_like(flux_map)

start, stop = 250, 260
for ii in range(start,stop):
    for jj in range(start,stop):
        for kk in range(Nz):
                if datacube.data[kk,jj,ii]<0: 
                    datacube.data[kk,jj,ii]=0;
        if flux_map_tmp[jj,ii] > 15*error:
            
            spec_tmp = datacube.data[:,jj,ii]
            spec_tmp = np.nan_to_num(spec_tmp)
            
            fit_params = Parameters()
            fit_params.add('amp_g1', value=0.03, min = 0.005, max= 0.1)
            fit_params.add('cen_g1', value=0, min = -300, max= 300)
            fit_params.add('wid_g1', value=200, min = 10, max = 300)
            p=1
            out = minimize(residual, fit_params, args=(x,p,), kws={'data': spec_tmp,'sigma': error})
            fit = residual(out.params, x, p)
            print('##')
            print('1gaussian_fit pixel:' + str(jj) + '-' + str(ii))
            report_fit(out, modelpars=p_true)
            plt.figure(figsize = (12,4))
            plt.plot(x, spec_tmp, label='data')
            plt.plot(x, data*0,':',color = 'black')
            plt.plot(x, fit, label='best fit')
            plt.xlabel('velocity [km/s]')
            plt.ylabel('flux [mJy]')
            plt.title('1Gaussian_model_pixel' + str(jj) + '-' + str(ii) )
            plt.legend()
            plt.show()
            print('##')
            print('1gaussian_fit')
            parvals = out.params.valuesdict()
            amplitude = parvals['amp_g1']
            stddev = parvals['wid_g1']
            mean = parvals['cen_g1']
            bic_1g = out.bic
            bic_1 = bic_1g
            bic_2 = 0
            while bic_1 > bic_2 and (bic_1 - bic_2) > 2.3:
                bic_1 = bic_2
                p = 2
                n = 2
                fit_params.add('amp_g' + str(n), value=0.01, min= 0.0025, max= 0.1)
                fit_params.add(name='peak_split', value=0, min=-500, max=500, vary=True)
                fit_params.add(name=('cen_g' + str(n)), expr='peak_split+cen_g1')
                fit_params.add(name=('wid_g' + str(n)), value=100, min =5, max= 200)
                out = minimize(residual, fit_params, args=(x,p,), kws={'data': spec_tmp, 'sigma':error})
                fit = residual(out.params, x, p)
                print('##')
                print(str(n) + 'gaussianfit' + 'pixel:' + str(jj) + '-' + str(ii))
                report_fit(out, modelpars=p_true)
                plt.figure(figsize = (12,4))
                plt.plot(x, spec_tmp, label='data')
                plt.plot(x, data*0,':',color = 'black')
                plt.plot(x, fit, label='best fit')
                plt.xlabel('velocity [km/s]')
                plt.ylabel('flux [mJy]')
                plt.title('2Gaussian_model_pixel' + str(jj) + '-' + str(ii) )
                plt.legend()
                plt.show()
                parvals = out.params.valuesdict()
                amplitude_2 = parvals['amp_g' + str(n)]
                stddev_2 = parvals['wid_g' + str(n)]
                mean_2 = parvals['cen_g' + str(n)]
                amplitude_1 = parvals['amp_g' + str(n-1)]
                stddev_1 = parvals['wid_g' + str(n-1)]
                mean_1 = parvals['cen_g' + str(n-1)]
                n += 1
                bic_2 = out.bic
                p += 1
            if bic_1g < bic_2:    
                vel_map[jj,ii] = mean
                vdisp_map[jj,ii] = stddev
                flux_map[jj,ii] = amplitude
            else:
                vel_map[jj,ii] = mean_2 + mean_1
                vdisp_map[jj,ii] = stddev_2 + stddev_1
                flux_map[jj,ii] = amplitude_2 + amplitude_1

#estimate galaxy size
#generate flux map

flux_map = np.nansum(datacube.data, axis = (0))
plt.imshow(flux_map, origin = 'lower')
plt.colorbar()



yy, xx = np.mgrid[:Ny, :Nx]

# Fit the data using astropy.modeling
model_init = models.Gaussian2D(amplitude=50, x_mean=250, y_mean=250, x_stddev=100, y_stddev=100, theta=45)
fit_model = fitting.LevMarLSQFitter()


model= fit_model(model_init, xx, yy, flux_map)



#plot best-fitting results

print("####################")
print("BEST FITTING RESULTS")
print("amplitude  = {:2f} mJy".format(model.amplitude.value))
print("x,y  = {:2f},{:2f} pixel".format(model.x_mean.value,model.y_mean.value))
print("x_stddev,y_stddev  = {:2f},{:2f} pixel".format(model.x_stddev.value,model.y_stddev.value))
print("####################")


plt.figure(figsize = (12,4))

plt.subplot(131)
plt.imshow(flux_map, origin = 'lower')
plt.subplot(132)
plt.imshow(model(xx,yy), origin = 'lower')
plt.subplot(133)
plt.imshow(flux_map-model(xx,yy), origin = 'lower')



#size pixel
cdelt1 = abs(datacube.header['CDELT1'])
print(cdelt1)

cdelt1_deg = cdelt1*u.deg
cdelt1_arcsec = cdelt1_deg.to('arcsec')
print(cdelt1_deg,cdelt1_arcsec)

redshift = 0.006775
arcsec_to_kpc =  p15.arcsec_per_kpc_proper(redshift)
print(arcsec_to_kpc)

pixel_size_kpc = cdelt1_arcsec/arcsec_to_kpc
print(pixel_size_kpc)



#estimate dynamical mass
FWHM = 307.56 *u.km/u.s
radius = 54.5*pixel_size_kpc
print('FWHM = {:.2f}'.format(FWHM))
print('radius  = {:.2f}'.format(radius))

Mdyn = FWHM**2*radius/K.G
print('Mdyn = {:e}'.format(Mdyn))
print('Mdyn = {:e}'.format(Mdyn.to('kg')))
print('Mdyn = {:e}'.format(Mdyn.to('M_sun')))


            

# #generete velocity maps 
# flux_map = np.nansum(datacube.data, axis = (0))
# vel_map = np.zeros_like(flux_map)
# vdisp_map2 = np.zeros_like(flux_map)
# vdisp_map = np.zeros_like(flux_map)

        

# for ii in range(Nx):
#     for jj in range(Ny):
#         if flux_map[jj,ii] > 0.0001*np.nanmax(flux_map):
#             for kk in range(Nz):
#                 if datacube.data[kk,jj,ii]< 3*error: #sostiture con 3/5 sigma
#                     datacube.data[kk,jj,ii]=0;
#             vel_map[jj,ii] = np.nansum((datacube.data[:,jj,ii]*velocity))/flux_map[jj,ii]
#             vdisp_map2[jj,ii] = np.nansum((datacube.data[:,jj,ii]*(velocity-vel_map[jj,ii])**2))/flux_map[jj,ii]

# vdisp_map = vdisp_map2**0.5   

# vel_map[flux_map < 0.0001*np.nanmax(flux_map)] = np.nan
# vdisp_map[flux_map < 0.0001*np.nanmax(flux_map)] = np.nan

# plt.figure(figsize = (12,4))

# plt.subplot(131)
# plt.imshow(flux_map, origin = 'lower', cmap = 'jet')
# plt.subplot(132)
# plt.imshow(vel_map, origin = 'lower', vmin = -100, vmax = 100, cmap ='jet')
# plt.subplot(133)
# plt.imshow(vdisp_map, origin = 'lower',  vmin = 0, vmax =500, cmap = 'jet')



# #Try to make a better map

# # #generete velocity maps 
# flux_map_tmp = np.nansum(datacube.data, axis = (0))

# flux_map = np.zeros_like(flux_map_tmp)
# vel_map = np.zeros_like(flux_map)
# vdisp_map = np.zeros_like(flux_map)




# for ii in range(Nx):
#     for jj in range(Ny):
#         if flux_map_tmp[jj,ii] > 0.1*np.nanmax(flux_map_tmp):
            
#             spec_tmp = datacube.data[:,jj,ii]
#             gaussian_int = models.Gaussian1D(amplitude= 0.01, mean= 0, stddev= 200)
#             fit_gaussian = fitting.LevMarLSQFitter()
#             gaussian = fit_gaussian(gaussian_int, velocity, spec_tmp)
            
#             vel_map[jj,ii] = gaussian.mean.value
#             vdisp_map[jj,ii] = gaussian.stddev.value
#             flux_map[jj,ii] = gaussian.amplitude.value


# plt.figure(figsize = (12,4))            
# plt.subplot(131)
# plt.imshow(flux_map, origin = 'lower', cmap = 'jet', vmin = 0, vmax= 0.0005)
# plt.subplot(132)
# plt.imshow(vel_map, origin = 'lower', vmin = -200, vmax = 200, cmap ='jet')
# plt.subplot(133)
# plt.imshow(vdisp_map, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')


# #extract the rotion curve

# vel_map[np.isnan(vel_map)] = 0.0
# vel_map_rot = scipy.ndimage.rotate(vel_map, 60.)
# plt.figure(figsize = (12,4))
# plt.imshow(vel_map_rot, origin = 'lower', vmin = -200, vmax = 200, cmap ='jet')

# Nyvel,Nxvel = vel_map_rot.shape
# x0_slit = 370
# dl_slit = 100
# rad = ((np.arange(0,Nyvel)-26)*pixel_size_kpc).value

# vel_curve = np.mean(vel_map_rot[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1)

# plt.figure(figsize = (12,4))
# plt.scatter(rad,vel_curve)
# plt.xlabel('radius [kpc]')
# plt.ylabel('velocity [km/s]')
# #plt.xlim(-2.5,2.5)
# plt.ylim(-200,200)


# idx_sel = np.where((rad>0) & (rad<3.0))
# rad_sel = rad[idx_sel]
# vel_curve_sel = vel_curve[idx_sel]

# plt.figure(figsize = (12,4))
# plt.scatter(rad_sel,vel_curve_sel, color = 'C0')
# plt.xlabel('radius [kpc]')
# plt.ylabel('velocity [km/s]')


# # Define model
# def vel_circular(x,Re=1.0,Mdyn=1e10):
    
#     R0 = 5.0
#     r = x
#     y = r/(2.*Re)
#     Ay = ( scipy.special.iv(0,y)*scipy.special.kv(0,y)-scipy.special.iv(1,y)*scipy.special.kv(1,y) )**0.5
    
#     V0 = 1e-3*((K.G.value*Mdyn*K.M_sun.value/(Re*K.kpc.value))**0.5)
     
#     B0 = (1.-np.exp(-R0/Re)*(1+R0/Re))**0.5

#     v_circ =  V0/B0*y*Ay*np.sqrt(2.0)    
    
#     return v_circ

# model_curve_init = vel_circular(Re=1.0,Mdyn=1e10)
# fit = LevMarLSQFitter()
# model_curv = fit(model_curve_init, rad_sel,vel_curve_sel)

# #print and plot best-fitting results

# print("####################")
# print("BEST FITTING RESULTS")
# print("radius  = {:2e} kpc".format(model_curv.Re.value))
# print("mass = {:2e} Msun".format(model_curv.Mdyn.value))
# print("####################")


# plt.scatter(rad_sel,vel_curve_sel, color = 'C0')
# plt.plot(rad_sel,model_curv(rad_sel),color ='C1')
# plt.xlabel('radius [kpc]')
# plt.ylabel('velocity [km/s]')
