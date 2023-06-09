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

# from scipy.optimize import curve_fit


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

# Qtoomre Parameter
def Q_toomre(k,c,E):
    G = K.G.to('kpc3 kg-1 s-2').value
    # c = 1* c *u.km.to('kpc')
    # Q = k * c / (np.pi * G * E)
    Q = k + c - np.log10(np.pi) - np.log10(G) - E
    return Q
def Q_toomre_err(k, c, E, k_err, c_err, E_err):
    G = K.G.to('kpc3 kg-1 s-2').value
    c = 1* c *u.km.to('kpc')
    c_err = 1 * c_err * u.km.to('kpc')
    Q = k * c / (np.pi * G * E)
    Q_err = Q * np.sqrt((k_err/k)**2+(c_err/c)**2+(E_err/E)**2)
    return Q_err

# Central method Derivate
def radial_derivative_c(velocity, radius):
    dV_dR = np.zeros_like(velocity)
    dV_dR[1:-1] = (velocity[2:] - velocity[:-2]) / (radius[2:] - radius[:-2])
    dV_dR[0] = (velocity[1] - velocity[0]) / (radius[1] - radius[0])
    dV_dR[-1] = (velocity[-1] - velocity[-2]) / (radius[-1] - radius[-2])
    return dV_dR

# Savgol method derivate
def radial_derivative_s(velocity, radius):
    window_size = 73 
    order = 6
    dV_dR = savgol_filter(velocity, window_size, order, deriv=1, delta=radius[1]-radius[0])
    return dV_dR


dV_dR_c = np.abs(radial_derivative_c(mean_vel, ell_rad_kpc.to('km').value)) #s^-1
dV_dR_s = np.abs(radial_derivative_s(mean_vel,ell_rad_kpc.to('km').value))  #s^-1

dV_dR_fit = radial_derivative_c(fit, ell_rad_kpc.to('km').value)

# Epicyclic Frequency
omega = mean_vel/ell_rad_kpc.to('km').value # s^-1
omega_err = omega * np.sqrt((m_v_err/mean_vel)**2 + (ell_rad_err_kpc/ell_rad_kpc)**2)
k_f = np.sqrt(2 * omega * (dV_dR_fit + omega))  #s^-2
k_f_err = np.sqrt(2) * omega_err

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
# E1 = M_out1 / (pixel_size_kpc.value)**2 #kg kpc^-2 
# E2 = M_out2 / (pixel_size_kpc.value)**2
# E3 = M_out3 / (pixel_size_kpc.value)**2

# Surface Density
E1 = np.full_like(flux, np.nan,dtype=np.float128)
E2 = np.full_like(flux,np.nan, dtype=np.float128)
E1[masknan] =  M_out1[masknan] / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)  # kg/kpc^2
E2 = M_out2 / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)
E3 =  M_out3 / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)
logE1 = logm1 - logpixel
logE2 = logm2 - logpixel
logE3 = logm3 - logpixel

# E1_err = M_out1_err / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)
# E2_err = M_out2_err / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)
# E3_err = M_out3_err / (beam_area_arcsec * pixel_area_kpc/pixel_area_arcsec)



#Mapping k_f (epicyclic frequency) and mean dispersion for the all image
# Define the center of the matrix and the distance from the center
cen = (225, 120)
d_min = 5
d_max = 8

# Create a meshgrid of x and y coordinates
xx, yy = np.meshgrid(np.arange(vel.shape[1]), np.arange(vel.shape[0]))

# Calculate the distance of each point from the center along the x and y axis
d_y = (yy - center[0])**2
d_x = (xx - center[1])**2 / (1 - ellipticity)**2

k_f_map = np.full_like(vel, np.nan)
k_f_err_map = np.full_like(vel,np.nan)
disp_map = np.full_like(vel, np.nan)
# disp_map_sub = 1 * disp                 #dispersion map - disp_map

for k in range(1, 111, 1):
    # Create a mask for points that lie within 2 ellipse of distance = 3 from eachother
    mask_1 = d_x + d_y <= d_max**2
    mask_2 = d_x + d_y > d_min**2
    mask_tot = np.logical_and(mask_1,mask_2)
    disp_map[mask_tot] = mean_disp[k-1]
    k_f_map[mask_tot] = k_f[k-1]
    k_f_err_map[mask_tot] = k_f_err[k-1]
    # disp_map_sub[mask_tot] = disp[mask_tot] - disp_map[mask_tot]
    

    d_min += 3
    d_max += 3

logk = np.log10(k_f_map)
disp = 1*disp *u.km
logdisp = np.log10(disp.to('kpc').value)


Q1_map = Q_toomre(logk, logdisp, logE1)
Q2_map = Q_toomre(logk, logdisp, logE2)
Q3_map = Q_toomre(logk, logdisp, logE3)
# Q-parameter with the 3 different alpha
Q1 = np.full_like(vel, np.nan)
Q2 = np.full_like(vel, np.nan)
Q3 = np.full_like(vel, np.nan)

Q1_map10 = 10**(Q1_map)
Q2_map10 = 10**(Q2_map)
Q3_map10 = 10**(Q3_map)

# Plot Q Toomre Parameter map

disp_map_sub = disp.value - disp_map
# Plotting the maps
plt.figure(figsize=(12,4))
plt.suptitle('Q-Toomre parameter map')
plt.subplot(131)
plt.imshow(Q1_map10, origin = 'lower', vmin= 0, vmax=2, cmap = 'jet')
plt.colorbar(label='Q1')
plt.title('Alpha = 0.8 M_sun/ K km s^-1 pc^2')
plt.subplot(132)
plt.imshow(Q2_map10, origin = 'lower',vmin= 0, vmax=2, cmap ='jet')
plt.colorbar(label='Q2')
plt.title('Alpha = 1.2 M_sun/ K km s^-1 pc^2')
plt.subplot(133)
plt.imshow(Q3_map10, origin = 'lower',vmin= 0, vmax=2, cmap = 'jet')
plt.colorbar(label='Q3')
plt.title('Alpha = 4.4 M_sun/ K km s^-1 pc^2')
plt.show()



plt.figure()
plt.imshow(Q1_map , origin = 'lower', vmin = -1, vmax = 0.5, cmap = 'jet')
plt.colorbar(label='')
plt.title('L_CO')
plt.show()


plt.figure()
plt.imshow(Q1_map10 , origin = 'lower', vmin = 0, vmax = 2, cmap = 'jet')
plt.colorbar()
plt.title('Q-toomre map')
plt.show


plt.figure()
plt.imshow(disp_map_sub, origin = 'lower', vmin = -50, vmax= 100, cmap = 'jet')
plt.colorbar(label='km/s')
plt.title('Dispersion map - Mean Dispersion map')
plt.show()

del(ell_rad, ell_rad_kpc)
# Q-toomre radial profile

Q1_map10[Q1_map10>10] = np.nan
Q2_map10[Q2_map10>10] = np.nan
Q3_map10[Q3_map10>10] = np.nan


Q1_radial = np.ndarray(74) 
Q2_radial = np.ndarray(74)
Q3_radial = np.ndarray(74)
Q1_err = np.ndarray(74)
Q2_err = np.ndarray(74)
Q3_err = np.ndarray(74)
ell_rad = np.ndarray(74)


# Define the center of the matrix and the distance from the center
center = (225, 120)
dist_min = 5
dist_max = 8

# Create a meshgrid of x and y coordinates
x, y = np.meshgrid(np.arange(vel.shape[1]), np.arange(vel.shape[0]))

# Calculate the distance of each point from the center along the x and y axis
dist_y = (y - center[0])**2
dist_x = (x - center[1])**2 / (1 - ellipticity)**2
contatore = 0

for k in range(1, 75, 1):
    # Create a mask for points that lie within 2 ellipse of distance = 3 from eachother
    mask1 = dist_x + dist_y <= dist_max**2
    mask2 = dist_x + dist_y > dist_min**2
    mask = np.logical_and(mask1,mask2)
    ell_rad[k-1] = (dist_max + dist_min) / 2
    ell_rad_err[k-1] = (dist_max - dist_min)/2
    Q1_radial[k-1] = np.nanmean(Q1_map10[mask])
    Q2_radial[k-1] = np.nanmean(Q2_map10[mask])
    Q3_radial[k-1] = np.nanmean(Q3_map10[mask])
    Q1_err = scipy.stats.sem(Q1_map10[mask], nan_policy='omit')
    Q2_err = scipy.stats.sem(Q2_map10[mask], nan_policy='omit')
    Q3_err = scipy.stats.sem(Q3_map10[mask], nan_policy='omit')
    
    dist_min += 3
    dist_max += 3

ell_rad_kpc = ell_rad*pixel_size_kpc
plt.figure(figsize = (12,5))
# plt.scatter(ell_rad_kpc, Q1, marker = '.', label = 'Alpha = 4.4 M_sun / K km s^-1 pc^2')
plt.plot(ell_rad_kpc, ((Q1_radial * 0) + 1),':', color='black', label ='Stability Cryterion')
plt.errorbar(ell_rad_kpc, Q1_radial, Q1_err, fmt='.', label= 'Alpha = 4.4 M_sun/ K km s^-1 pc^2')
plt.errorbar(ell_rad_kpc, Q2_radial, Q2_err, fmt='.', label= 'Alpha = 1.2 M_sun/ K km s^-1 pc^2')
plt.errorbar(ell_rad_kpc, Q3_radial, Q3_err, fmt='.', label= 'Alpha = 0.8 M_sun/ K km s^-1 pc^2')
plt.xlabel('R[kpc]')
plt.ylabel('Q')
plt.title('Q Toomre parameter in the different ellipses (Q(r)= <Q(r)>)')
plt.legend()
plt.show()