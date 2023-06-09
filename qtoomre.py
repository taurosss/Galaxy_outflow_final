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

flux = fits.open(path + '/moments_map/NGC6810/flux_crop.fits')[0]
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
flux = flux.data
vel = vel.data
disp = disp.data
velerr = velerr.data
disperr = disperr.data

#----------------------------------------------------------------------------
# Binning of the data
# vel1 = rebin(vel,(225,120))
# disp1 = rebin(disp,(225,120))
# disperr1 = rebin(disperr, (225,120))

# Ny,Nx = 225, 120
# yy, xx = np.mgrid[:Ny, :Nx]
# x0, y0 = 60, 112.5
# yy = yy -y0
# xx = xx - x0
Ny,Nx = 450, 240
yy, xx = np.mgrid[:Ny, :Nx]
x0, y0 = 120, 225
yy = yy -y0
xx = xx - x0

xbin1 = xx.flatten()
ybin1 = yy.flatten()
velbin1 = vel.flatten()
dispbin1 = disp.flatten()

xbin = np.delete(xbin1, np.isnan(velbin1))
ybin = np.delete(ybin1, np.isnan(velbin1))
velbin = np.delete(velbin1, np.isnan(velbin1))
velerrbin = np.full_like(velbin, np.abs(velbin*0.01))
# DD = np.delete(D, np.isnan(C))
dispbin = np.delete(dispbin1, np.isnan(velbin1))
disperrbin = np.full_like(dispbin, np.abs(dispbin*0.01))
# FF = np.delete(E, np.isnan(C))

#----------------------------------------------------------------------------
# size pixel
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

#----------------------------------------------------------------------------
# estimate dynamical mass
FWHM = 308.3852 *u.km/u.s
radius = 54.5*pixel_size_kpc
print('FWHM = {:.2f}'.format(FWHM))
print('radius  = {:.2f}'.format(radius))

Mdyn = FWHM**2*radius/K.G
print('Mdyn = {:e}'.format(Mdyn))
print('Mdyn = {:e}'.format(Mdyn.to('kg')))
print('Mdyn = {:e}'.format(Mdyn.to('M_sun')))

#----------------------------------------------------------------------------
#Generate by-symmetric version of kinematics measurements
velsym = symmetrize_velfield(xbin1, ybin1, velbin1, sym = 3, pa = 85)
t = 0
velsym_m = np.ndarray((450,240))
for jj in range(450):
    for ii in range(240):
        velsym_m[jj,ii] = velsym[t]
        t +=1
        
plt.figure()
plt.imshow(velsym_m, origin = 'lower', vmin = -350, vmax = 350, cmap ='jet')
plt.colorbar()
plt.title('Velocity map symmetrized')

#----------------------------------------------------------------------------
#Velocity and dispersion mean Slit Version

Nyvel,Nxvel = velsym_m.shape
x0_slit = 120
dl_slit = 3
 # rad = ((np.arange(0,Nyvel)-240)*pixel_size_kpc).value
rad = (np.arange(-225, 225, 1) * pixel_size_kpc).value

# velsym_m[np.isnan(velsym_m)]=0.0
vel_curve = np.nanmean(vel[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1)
err_vel_curve = scipy.stats.sem(vel[:, x0_slit-dl_slit:x0_slit+dl_slit], axis=1, nan_policy='omit')
# err_vel_curve = (np.nanmax(vel[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1) - np.nanmin(vel[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1))/2
disp_curve= np.nanmean(disp[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1)
err_disp_curve = scipy.stats.sem(disp[:, x0_slit-dl_slit:x0_slit+dl_slit], axis=1, nan_policy='omit')
# err_disp_curve = (np.nanmax(disp[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1) - np.nanmin(disp[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1))/2

plt.figure(figsize = (20,7))
# plt.errorbar(rad,np.abs(vel_curve), err_vel_curve, fmt='.',label='velocity')
# plt.errorbar(rad,np.abs(disp_curve), err_disp_curve, fmt='.', label='dispersion')
plt.scatter(rad,np.abs(vel_curve), marker='.',label='velocity')
plt.scatter(rad,np.abs(disp_curve), marker='.', label='dispersion')
plt.xlabel('radius [kpc]')
plt.ylabel('velocity [km/s]')
#plt.xlim(-2.5,2.5)
plt.ylim(0,300)
plt.legend()
# plt.scatter(rad,np.abs(disp_curve), color='red', marker='.')
plt.title('Velocity and Dispersion velocity mean in a 6 pixel slit')


plt.figure()
err_v_d = np.sqrt(((vel_curve * err_vel_curve)**2 + (disp_curve * err_disp_curve)**2)/(disp_curve**2))
plt.scatter(rad, np.abs(vel_curve/disp_curve), marker='.')
# plt.errorbar(rad,np.abs(vel_curve/disp_curve), err_v_d, fmt='.')
plt.ylim(0,20)
plt.xlabel('radius [kpc]')
plt.ylabel('Velocity/Velocity Dispersion')
plt.title('Velocity/Velocity Dispersion with the 6 pixel slit')

del(vel_curve, disp_curve, err_vel_curve, err_disp_curve)
#----------------------------------------------------------------------------
# simply rotate velocity and dispersion 
PA = -5
vel1 = vel + 0.0000001
vel1[np.isnan(vel1)] = 0.0
vel_rot = scipy.ndimage.rotate(vel1, PA)
disp1 = disp + 0.0000001
disp1[np.isnan(disp)] = 0.0
disp_rot = scipy.ndimage.rotate(disp1, PA)
plt.figure(figsize = (8,4))
plt.subplot(121)
plt.imshow(vel_rot, origin = 'lower', vmin = -350, vmax = 350, cmap ='jet')
plt.colorbar(label= 'velocity [km/s]')
plt.title('Rotated velocity map')
plt.subplot(122)
plt.imshow(disp_rot, origin = 'lower', vmin = 0, vmax = 200, cmap ='jet')
plt.colorbar(label= 'velocity dispersion [km/s]')
plt.title('Rotated dispersion map')

del(vel_rot, disp_rot, vel1, disp1)
Q = 0.5
e = 1 - Q
inc = np.arctan(e)

#----------------------------------------------------------------------------
#Rotate with different angles and extract v/sigma (y_pixel)
printplot = False
v_s = np.ndarray([7, 450])
v_s_err = np.ndarray([7,450])
disp = np.abs(disp)
disp_rot = np.abs(disp)
veli = vel/np.sin(inc)
vel_rot = vel/np.sin(inc)
# Making the medium of a velocity and dispersione for every raw with dimension 5 pixel 
cont = 0
for PA in range(-5, -96, -15):
    # vel_rot = scipy.ndimage.rotate(veli, 0, reshape = False)  #trying to rotate only vdisp
    disp_rot[np.isnan(disp_rot)] = 0.
    disp_rot = scipy.ndimage.rotate(disp_rot, PA, reshape = False)
    disp_rot[np.where(disp_rot ==0.)] = np.nan
    
    vel_curve = np.nanmean(vel_rot[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1)
    disp_curve = np.nanmean(disp_rot[:,x0_slit-dl_slit:x0_slit+dl_slit],axis = 1)
    err_vel_curve = scipy.stats.sem(vel[:, x0_slit-dl_slit:x0_slit+dl_slit], axis=1, nan_policy='omit')
    err_disp_curve = scipy.stats.sem(disp[:, x0_slit-dl_slit:x0_slit+dl_slit], axis=1, nan_policy='omit')    
   
    v_s[cont] = np.abs(vel_curve/disp_curve)
    v_s_err[cont] = np.sqrt(((vel_curve * err_vel_curve)**2 + (disp_curve * err_disp_curve)**2)/(disp_curve**2))
    cont +=1
    if printplot == True:
        plt.figure(figsize = (8,4))
        plt.subplot(121)
        plt.imshow(vel_rot, origin = 'lower', vmin = -350, vmax = 350, cmap ='jet')
        plt.colorbar(label= 'velocity [km/s]')
        plt.title('Rotated velocity map')
        plt.subplot(122)
        plt.imshow(disp_rot, origin = 'lower', vmin = 0, vmax = 200, cmap ='jet')
        plt.colorbar(label= 'velocity dispersion [km/s]')
        plt.title('Rotated dispersion map')

del(cont)


#----------------------------------------------------------------------------
# Plotting v/sigma for every angle 
y = np.arange(-225,225,1)
rad = ((np.arange(0,Nyvel)-225)*pixel_size_kpc).value
for i in range(0,7):
    
    plt.figure(figsize = (8,4))
    # plt.errorbar(rad, v_s[i], v_s_err[i], label= str(i*15) + '°', fmt='.')
    plt.scatter(rad, v_s[i], label = str(i*15) + '°', marker = '.')
    # plt.plot(y, v_s[1], label= '15°')
    # plt.plot(y, v_s[2], label='30°')
    # plt.plot(y, v_s[3], label= '45°')
    # plt.plot(y, v_s[4], label='60°')
    # plt.plot(y, v_s[5], label= '75°')
    # plt.plot(y, v_s[6], label='90°')
    plt.xlabel('rad[kpc]')
    plt.ylabel('velocity/sigma')
    plt.legend()
    plt.title('V vs Sigma in a Slit Rotating the Galaxy of 15°')
    plt.ylim(0,20)
    plt.show()

del(v_s, v_s_err)
#----------------------------------------------------------------------------
# Velocity/sigma in radial 2d-cone 
# Define the center of the galaxy and the angle of the circular crown
center = (225, 120)  # center of the matrix
angle = 15  # angle of the circular crown in degrees

# Define the range of radii to consider
min_radius = 0


# Create a circular mask for the current radius
y, x = np.ogrid[:450, :240]
dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
# Create a mask for the circular crown
angles = np.arctan2(y-center[0],x-center[1])*180/np.pi

for PA in range(0,180,30):
    velocity_means = []
    disp_means = []
    vel_errors = []
    disp_errors = []
    v_s_err = []
    v_s = []
    radius_pixel = []
    if PA == 0 or PA == 150:
        max_radius = 120
    elif PA == 30 or PA == 120:
        max_radius = 160
    elif PA == 60 or PA == 90:
        max_radius = 200
    
    radii = np.arange(min_radius, max_radius+1)
    #Iterate over the radii
    for radius in radii:

        mask = dist_from_center <= radius
        # angles[angles<0] +=360
        # angles_mask = (angles>= angle-180/np.pi*15) & (angles<= angle+180/np.pi*15)
        angles_mask = (angles >= (-PA -30)) & (angles<= -PA)
        mask = mask & angles_mask

        # Apply the mask to the velocity field
        # masked_velocity = np.ma.array(veli, mask=np.logical_not(mask))
        masked_velocity = veli[mask]
        # masked_dispersion = np.ma.array(disp, mask=np.logical_not(mask))
        masked_dispersion = disp[mask]

        # Calculate the mean velocity within the circular crown
        velocity_mean = np.nanmean(masked_velocity)
        vel_err = scipy.stats.sem(masked_velocity, nan_policy='omit')
        disp_mean = np.nanmean(masked_dispersion)
        disp_err = scipy.stats.sem(masked_dispersion, nan_policy='omit')

        # Append the velocity mean and radius to the array
        velocity_means.append(velocity_mean)
        disp_means.append(disp_mean)
        radius_pixel.append(radius)
        # vel_errors.append(vel_err)
        # disp_errors.append(disp_err)
        v_s.append(np.abs(velocity_mean/disp_mean))
        v_s_err.append(np.sqrt(((velocity_mean * vel_err)**2 + (disp_mean * disp_err)**2)/(disp_mean**2)))
        
        
        # plt.figure()
        # plt.imshow(vel, cmap='jet', vmin = -350, vmax = 350)
        # # Overlay the circular crown mask on the plot
        # plt.imshow(mask, cmap='gray', alpha=0.5)
        # # Add a circle to indicate the radius of the circular crown
        # circle = plt.Circle((120,225), radius, color='red', fill=False)
        # plt.gcf().gca().add_artist(circle)
        # plt.colorbar()
        # plt.title('Circular crown')
        # plt.show()
        
    radius_kpc = radius_pixel * pixel_size_kpc
    plt.figure(figsize = (12,4))
    plt.scatter(radius_kpc, v_s, label= 'v vs sigma', marker='.')
    plt.xlabel('R [kpc]')
    plt.ylabel('v/s')
    plt.title('v/s in the circular crown ' +str(PA) + '-' + str(PA+30) + '°')
    plt.legend()
    plt.show()
    del(angles_mask, mask, velocity_means,disp_means, disp_errors, vel_errors, v_s, v_s_err, radius_pixel)



# R = np.arange(0, 120,1)
# cont = 0 #contator for v_s = velocity/sigma
# mm = np.tan(10*np.pi/180)
# v_s = np.ndarray([8, 120])  #Velocity/sigma matrix
# v_s_err = np.ndarray([8, 120]) 
# xx_n = xx/ np.sin(inc)
# for PA in range(0, 360, 45):
#     vel_r = scipy.ndimage.rotate(veli, 0, reshape = False)
#     disp_r = scipy.ndimage.rotate(disp, PA, reshape = False)
#     v = vel_r[int(vel.shape[0]/2):vel.shape[0], int(vel.shape[1]/2):vel.shape[1]]         #first quadrant
#     s = disp_r[int(disp.shape[0]/2):disp.shape[0], int(disp.shape[1]/2):disp.shape[1]]
#     test = np.ndarray((8,120))
#     v_m = np.zeros([len(np.diag(v))])   #mean velocity 1darray
#     s_m = np.zeros([len(np.diag(s))])   #mean sigma velocity 1darray
#     err_v = np.zeros([len(np.diag(v))])   #mean velocity 1darray
#     err_s = np.zeros([len(np.diag(s))])
#     v_t = []
#     s_t = []
#     # plt.figure(figsize = (12,4))
#     # plt.scatter(R, s_m, label= '0-45˚')
#     # plt.xlabel('R [pixel unit]')
#     # plt.ylabel('sigma')
#     # plt.title('sigma in different circular 2d-cone')
#     # plt.legend()
#     # plt.show()       
#     for jj in range(225):
#         for ii in range(120):
#             if jj < mm *ii:
#                 for R in range(0, len(np.diag(v)), 3):
#                     if jj**2 + ii**2 > R**2 and jj**2 + ii**2 < (3 + R)**2:
#                         if np.isnan(v[jj,ii]) == False: 
#                             # v_m[R] += v[jj,ii] 
#                             # s_m[R] += s[jj,ii]
#                             v_t.append(v[jj,ii])
#                             s_t.append(v[jj,ii])
#                 v_m[R] /= np.mean(v_t)
#                 err_v[R] = scipy.stats.sem(v_t, nan_policy='omit')
#                 s_m[R] /= np.mean(s_t)
#                 err_s[R] = scipy.stats.sem(s_t, nan_policy= 'omit')
                    
#     v_s[cont] = np.abs(v_m/s_m) 
#     v_s_err[cont] =  
#     cont +=1
    
# R = np.arange(0, len(np.diag(v)),1)

# # plt.figure(figsize = (12,4))
# # plt.plot(R, v_s[0], label= '0-45˚')
# # plt.plot(R, v_s[1], label= '45-90˚')
# # plt.plot(R, v_s[2], label= '90-135˚')
# # plt.plot(R, v_s[3], label= '135-180˚')
# # plt.plot(R, v_s[4], label= '180-225˚')
# # plt.plot(R, v_s[5], label= '225-270˚')
# # plt.plot(R, v_s[6], label= '270-315˚')
# # plt.plot(R, v_s[7], label= '315-360˚')
# # plt.xlabel('R [pixel unit]')
# # plt.ylabel('velocity/sigma')
# # plt.title('velocity/sigma in different circular 2d-cone')
# # plt.legend()
# # plt.show()

# plt.figure(figsize = (12,4))
# plt.scatter(R, v_s[0], label= '0-45˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('velocity/sigma')
# plt.title('velocity/sigma in different circular 2d-cone')
# plt.legend()
# plt.show()
    
# plt.figure(figsize = (12,4))
# plt.scatter(R, v_s[1], label= '45-90˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('velocity/sigma')
# plt.title('velocity/sigma in different circular 2d-cone')
# plt.legend()
# plt.show()

# plt.figure(figsize = (12,4))
# plt.scatter(R, v_s[2], label= '90-135˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('velocity/sigma')
# plt.title('velocity/sigma in different circular 2d-cone')
# plt.legend()
# plt.show()

# plt.figure(figsize = (12,4))
# plt.scatter(R, v_s[3], label= '135-180˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('velocity/sigma')
# plt.title('velocity/sigma in different circular 2d-cone')
# plt.legend()
# plt.show()

# plt.figure(figsize = (12,4))
# plt.scatter(R, v_s[4], label= '180-225˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('velocity/sigma')
# plt.title('velocity/sigma in different circular 2d-cone')
# plt.legend()
# plt.show()

# plt.figure(figsize = (12,4))
# plt.scatter(R, v_s[5], label= '225-270˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('velocity/sigma')
# plt.title('velocity/sigma in different circular 2d-cone')
# plt.legend()
# plt.show()

# plt.figure(figsize = (12,4))
# plt.scatter(R, v_s[6], label= '270-315˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('velocity/sigma')
# plt.title('velocity/sigma in different circular 2d-cone')
# plt.legend()
# plt.show()
    
# plt.figure(figsize = (12,4))
# plt.scatter(R, v_s[7], label= '315-360˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('velocity/sigma')
# plt.title('velocity/sigma in different circular 2d-cone')
# plt.legend()
# plt.show()
    

    
# plt.figure(figsize = (12,4))
# plt.scatter(R, s_m, label= '0-45˚', marker='.')
# plt.xlabel('R [pixel unit]')
# plt.ylabel('sigma')
# plt.title('sigma in different circular 2d-cone')
# plt.legend()
# plt.show()
#----------------------------------------------------------------------------
#Velocity/Dispersion along ellipses


#One possible way to treat the parity of the velocity field
# vel = vel[int(vel.shape[0]/2):vel.shape[0], :vel.shape[1]]         #the semiplane with y>0, beacuse if we take the mean in all the ellipse the velocity has opposite sign from positive to negative y
# disp = disp[int(disp.shape[0]/2):disp.shape[0], :disp.shape[1]]
#The other possibility is to take the absolute value of the velocity in the mean
mean_vel = np.ndarray(44) 
mean_disp = np.ndarray(44)
m_v_err = np.ndarray(44)
m_d_err = np.ndarray(44)
v_d_err = np.ndarray(44)
ell_rad = np.ndarray(44)
# Define the center of the matrix and the distance from the center
center = (225, 120)
dist_min = 5
dist_max = 10

# Define the ellipticity of the ellipse
ellipticity = e

# Create a meshgrid of x and y coordinates
x, y = np.meshgrid(np.arange(vel.shape[1]), np.arange(vel.shape[0]))

# Calculate the distance of each point from the center along the x and y axis
dist_y = (y - center[0])**2
dist_x = (x - center[1])**2 / (1 - ellipticity)**2
contatore = 0

for k in range(1, 45, 1):
    # Create a mask for points that lie on the ellipse
    mask1 = dist_x + dist_y <= dist_max**2
    mask2 = dist_x + dist_y > dist_min**2
    mask = np.logical_and(mask1,mask2)
    ell_rad[k-1] = (dist_max + dist_min) / 2
    
    # Calculate the mean velocity along the ellipse
    mean_velocity = np.nanmean(np.abs(vel[mask]))   #capire se con questo metodo devo comunque correggere per l'inclinazione
    mean_dispersion = np.nanmean(np.abs(disp[mask]))
    mean_vel[k-1]=(mean_velocity)
    mean_disp[k-1]=(mean_dispersion)
    sem_vel = scipy.stats.sem(vel[mask], nan_policy='omit')
    sem_disp = scipy.stats.sem(disp[mask], nan_policy='omit')
    m_v_err[k-1] = sem_vel
    m_d_err[k-1] = sem_disp
    
    # plt.figure()
    # plt.imshow(vel, origin = 'lower', vmin = -350, vmax = 350, cmap ='jet')
    # plt.title('Last 2 Ellipses plotted on the velocity map' + str(contatore))
    # plt.colorbar()
    # plt.contour(x, y, mask1, colors='r', linewidths=1)
    # plt.contour(x, y, mask2, colors='b', linewidths=1)
    # plt.show()
    # contatore += 1
    # print(vel[mask], mean_velocity)

    dist_min += 5
    dist_max += 5
    
v_d_err = np.sqrt(np.sqrt(((mean_vel * m_v_err)**2 + (mean_disp * m_d_err)**2)/ (mean_disp**2)))
ell_rad_kpc = ell_rad*pixel_size_kpc
#----------------------------------------------------------------------------
#PLOTTING V/S, V_mean, S_mean
plt.figure(figsize = (12,4))
# plt.scatter(ell_rad_kpc, mean_vel/mean_disp, label= 'v vs sigma', marker='.')
plt.errorbar(ell_rad_kpc, mean_vel/mean_disp, v_d_err, fmt='.', label='v vs sigma (Ellipses Version)')
plt.xlabel('R [kpc]')
plt.ylabel('v/s')
plt.title('v/s in the different ellipses')
plt.legend()
plt.show()

plt.figure(figsize = (12,4))
plt.errorbar(ell_rad_kpc, mean_vel, m_v_err, fmt = '.', label = 'Mean velocity in the different ellipses')
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
#Qtoomre Parameter
def Q_toomre(k,c,E):
    G = K.G.value
    Q = k * c / (np.pi * G * E)
    return Q

#Central method Derivate
def radial_derivative_c(velocity, radius):
    dV_dR = np.zeros_like(velocity)
    dV_dR[1:-1] = (velocity[2:] - velocity[:-2]) / (radius[2:] - radius[:-2])
    dV_dR[0] = (velocity[1] - velocity[0]) / (radius[1] - radius[0])
    dV_dR[-1] = (velocity[-1] - velocity[-2]) / (radius[-1] - radius[-2])
    return dV_dR

vel_rot_i = mean_vel/np.sin(inc)
dV_dR_c = radial_derivative_c(mean_vel, ell_rad_kpc.value)

#Savgol method derivate
def radial_derivative_s(velocity, radius):
    window_size = 23 
    order = 3 
    dV_dR = savgol_filter(velocity, window_size, order, deriv=1, delta=radius[1]-radius[0])
    return dV_dR

dV_dR_s = np.abs(radial_derivative_s(mean_vel, ell_rad_kpc.value))

dV_dR_2c = np.abs(radial_derivative_c(vel_rot_i, ell_rad_kpc.value))
dV_dR_2s = np.abs(radial_derivative_s(vel_rot_i,ell_rad_kpc.value))

#Epicyclic Frequency
omega = vel_rot_i/ell_rad_kpc.value
k_f = 2 * omega * (dV_dR_2c + omega)  


#Conversion ratio L_CO --> L_H
alpha = 30 * K.M_sun.value / K.L_sun.value

#Calculate the Totale Flux 
# Identify the spectral range that contains the CII emission line
cii_range = np.where((velocity >= -250) & (velocity <= 300))[0]

# # Define a function to fit the continuum
# def continuum_fit(x, a, b, c):
#     return a * x**2 + b * x + c

#location of the target
# location of the target
x_0,y_0 = 250, 250
# size of the square aperture 
dl = 100
#extract the spectrum

spectrum = np.nansum(datacube.data[:,y_0-dl:y_0+dl,x_0-dl:x_0+dl],axis = (1,2))
# # Fit the continuum
# popt, pcov = curve_fit(continuum_fit, velocity, spectrum)


# # Subtract the continuum from the spectrum
# residual_spectrum = spectrum - continuum_fit(velocity, *popt)

# Integrate the spectrum
co_flux = scipy.integrate.trapz(spectrum[cii_range], dx=dv)
co_flux /= 10**3 #From mJy to Jy km s^-1
# cii_flux /= 10**-26 #From Jy to W

#Galaxy Distance
D = 27.1e6#pc
# parsec = 3.0857e16 #m
# D = D * parsec #From parsec to m
D = 27.1 #Mpc


freq = frequency_mean.value #*10**9
# freq = 1/ freq  #from hertz to second 
#CII Luminosity
L_c = 1.04e-3 * co_flux * D**2 * freq #This gives the Luminosity in L_sun

#Gas Mass
M_gas = alpha * L_c

#Surface density
E = M_gas/pixel_size_kpc.value


#Fabian Method
L_CO = 3.25e7 * co_flux / freq**2 / (1+redshift)**-3

alpha1 = 0.8 * K.M_sun #/(K km s^-1 pc^2)
alpha2 = 1.2 * K.M_sun 
alpha3= 4.4 * K.M_sun
M_out1 = alpha1 * L_CO #M_sun
M_out2 = alpha2 * L_CO
M_out3 = alpha3 * L_CO
E1 = M_out1 / pixel_size_kpc.value #kg kpc^-2
# E1 /= K.M_sun.value
E2 = M_out2 / pixel_size_kpc.value
# E2 /= K.M_sun.value
E3 = M_out3 / pixel_size_kpc.value
# E3 /= K.M_sun.value

Q1 = Q_toomre(k_f, mean_disp, E1)
Q2 = Q_toomre(k_f, mean_disp, E2)
Q3 = Q_toomre(k_f, mean_disp, E3)

#Plot Q Toomre Parameter
plt.figure(figsize = (12,4))
plt.scatter(ell_rad_kpc, Q1, marker = '.', label = 'Alpha = 0.8 M_sun / K km s^-1 pc^2')
plt.xlabel('R[kpc]')
plt.ylabel('Q')
plt.title('Q Toomre parameter in the different ellipses')
plt.legend()
plt.show()

plt.figure(figsize = (12,4))
plt.scatter(ell_rad_kpc, Q1, marker = '.', label = 'Alpha = 1.2 M_sun / K km s^-1 pc^2')
plt.xlabel('R[kpc]')
plt.ylabel('Q')
plt.title('Q Toomre parameter in the different ellipses')
plt.legend()
plt.show()

plt.figure(figsize = (12,4))
plt.scatter(ell_rad_kpc, Q1, marker = '.', label = 'Alpha = 4.4 M_sun / K km s^-1 pc^2')
plt.xlabel('R[kpc]')
plt.ylabel('Q')
plt.title('Q Toomre parameter in the different ellipses')
plt.legend()
plt.show()


# # Plot the original spectrum and the residual spectrum
# plt.plot(velocity, spectrum, label='Original Spectrum')
# # plt.plot(velocity, residual_spectrum, label='Residual Spectrum')
# plt.xlabel('Velocity [km/s]')
# plt.ylabel('Flux [mJy]')
# plt.legend()
# plt.show()
