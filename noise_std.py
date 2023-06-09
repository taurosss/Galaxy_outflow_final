#import python libraries
import numpy as np 
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.modeling import models, fitting
from scipy.ndimage import gaussian_filter
import os 
import time
#from astropy.nddata import Cutout2D
from scipy import ndimage
import astropy.constants as K
import astropy.units as u
from astropy.cosmology import Planck15 as p15
import scipy.ndimage
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from lmfit import minimize, Parameters, report_fit


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


#data/power response
noise_cube = datacube.data / datacube_antenna.data

#Choosing an empty region
x0, y0 = 294, 143
dl = 20
noise = noise_cube[:,y0-dl:y0+dl,x0-dl:x0+dl]
for ii in range(dl):
    for jj in range(dl):
            noise[0, jj, ii] = 0
#requency_window = frequency[:,y0-dl:y0+dl,x0-dl:x0+dl]
noise_spectrum = np.nansum(noise,axis = (0))
#number of pixel selected
N = (2*dl)**2
for ii in range(40):
    for jj in range(40):
        rms = np.sum((noise_spectrum[ii,jj]**2))
        
rms = np.sqrt(rms/N)


        
print("rms  = {:2f} mJy".format(rms))



plt.figure(figsize = (12,4))
plt.plot( frequency, noise[:,14,14], label = 'data')
plt.plot( frequency, frequency *0,':',color = 'black')

plt.xlabel('frequency [Ghz]')
plt.ylabel('flux [mJy]')
plt.title('Noise fit')
plt.legend()
plt.show()

