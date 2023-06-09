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

"""
FUNCTIONS
"""

##Making the Spiral grid for fitting
def invers_spiral(A):
    return A[::-1]                #inverting the array, so it starts from the center 

def spiral_mat_to_vect(A):
    v = []
    while(A.size != 0):
        v.append(A[0,:])
        A = A[1:,:].T[::-1]
    return np.concatenate(v)

def spiral_vect_to_mat(v):
    L = int(np.sqrt(v.size)) # lunghezza del pezzo da aggiungere
    l = L
    A = np.zeros((L,L))
    i = 3 # parto da 3 per fare in modo che la coordinata x aumenti al secondo step
    x = 0 # coordinata x del nuovo pezzo
    y = 0 # coordinata y del nuovo pezzo
    
    A[x,y:l] = v[0:l]
    A = A.T[::-1]
    v = v[l:len(v)]

    while(v.size != 0):
        i += 1 # Ad ogni step ruoto e riempio la prima riga della matrice
        if i % 2 == 0: # Ogni due rotazioni si accorcia la lunghezza l
            l -= 1
        if (i + 1) % 4 == 0: # Ogni 4 rotazioni x aumenta
            x += 1
        if i % 4 == 0: # Ogni 4 rotazioni y aumenta con un ritardo di 1 step rispetto a x
            y += 1
        A[x,y:y+l] = v[0:l]
        A = A.T[::-1]
        v = v[l:len(v)]
        
    for rotations in range(i % 4): # Faccio le rotazioni che mancano per rimettere la matrice nel verso giusto
        A = A.T[::-1]
        
    return A

## Multi-gaussians model

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



"""
MAIN
"""
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
dv = velocity[0]-velocity[1]

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




## RMS DETERMINATION WITH THE POWER RESPONSE 

#data/power response
noise_cube = datacube.data / datacube_antenna.data

#Choosing an empty region
x0, y0 = 294, 143
dl = 20
noise = noise_cube[:,y0-dl:y0+dl,x0-dl:x0+dl]
error = np.std(noise[1:,:,:])

print("rms  = {:2f} mJy".format(error))
print("####################")




#2plot: velocity - spectrum
plt.figure(figsize = (12,4))
plt.plot(velocity, spectrum, label = 'data')
plt.plot(velocity,frequency*0,':',color = 'black')
plt.xlabel('velocity [km/s]')
plt.ylabel('flux ')
plt.title('Totale Spectrum (function of velocity)')
plt.legend()
plt.show()



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


    
#Generating moments map
mask_cube = np.where(datacube.data > 3*error, datacube.data, np.nan)
M0 = np.nansum(mask_cube, axis = (0))*dv
M1 = np.nansum(mask_cube[:,:,:]*velocity[:,np.newaxis,np.newaxis], axis=0)*dv / M0
# avoid division by 0 or neg values in sqrt
# thr = np.nanpercentile(M0[np.where(M0>0)],0.01)
thr = 3*error
M0[np.where(M0<thr)]=np.nan
M2 = np.sqrt(np.nansum(np.power(datacube.data[:, :, :] * (velocity[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :]),2), axis=0) * dv / M0 )   

# #generete velocity maps 
moment0 = np.nansum(datacube.data, axis = (0))
#Creating the matrix-coordinates that would be spiralized
mat_tmp =  np.empty([Nx, Ny], dtype='<U7')
for i in range(Nx):
    for j in range(Ny):
        mat_tmp[i,j] = (str(j) + '-' + str(i))  
        
#Spiralizing the matrix
spiral_vect = spiral_mat_to_vect(mat_tmp)
spiral_vect = invers_spiral(spiral_vect)

mask_cube = datacube.data>4.3*error
        
flux_map_tmp = np.nansum(datacube.data, axis = (0))
flux_map = np.full_like(flux_map_tmp,np.nan)
vel_map = np.full_like(flux_map,np.nan)
vdisp_map = np.full_like(flux_map,np.nan)

fit_params1g = Parameters()
fit_params1g.add('amp_g1', value=0.03, min = 0.005, max= 0.1)
fit_params1g.add('cen_g1', value=134, min = -300, max= 300)
fit_params1g.add('wid_g1', value=20, min = 10, max = 300)

fit_params2g = Parameters()
fit_params2g.add('amp_g1', value=0.03, min = 0.005, max= 0.1)
fit_params2g.add('cen_g1', value=134, min = -300, max= 300)
fit_params2g.add('wid_g1', value=19, min = 10, max = 300)
fit_params2g.add('amp_g2', value=0.025, min = 0.005, max= 0.1)
fit_params2g.add(name='peak_split', value=50, min=-200, max=250, vary=True)
fit_params2g.add(name=('cen_g2'), expr='peak_split+cen_g1')
fit_params2g.add('wid_g2', value=20, min = 10, max = 300)

fit_params3g = Parameters()
fit_params3g.add('amp_g1', value=0.03, min = 0.005, max= 0.1)
fit_params3g.add('cen_g1', value=134, min = -300, max= 300)
fit_params3g.add('wid_g1', value=150, min = 10, max = 300)
fit_params3g.add('amp_g2', value=0.025, min = 0.005, max= 0.1)
fit_params3g.add(name='peak_split', value=50, min=-200, max=250, vary=True)
fit_params3g.add(name=('cen_g2'), expr='peak_split+cen_g1')
fit_params3g.add('wid_g2', value=20, min = 10, max = 300)
fit_params3g.add('amp_g3', value=0.02, min = 0.005, max= 0.1)
fit_params3g.add(name=('cen_g3'), expr='peak_split+cen_g2')
fit_params3g.add('wid_g3', value=20, min = 10, max = 300)

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

fit_params3gx = Parameters()
fit_params3gx.add('amp_g1', value=0.035, min = 0.005, max= 0.1)
fit_params3gx.add('cen_g1', value=M1[250,250], min = -300, max= 300)
fit_params3gx.add('wid_g1', value=60, min = 5, max = 200)
fit_params3gx.add('amp_g2', value=0.02, min = 0.005, max= 0.1)
fit_params3gx.add(name='peak_split', value=50, min=-200, max=250, vary=True)
fit_params3gx.add(name=('cen_g2'), expr='peak_split+cen_g1')
fit_params3gx.add('wid_g2', value=20, min = 5, max = 200)
fit_params3gx.add('amp_g3', value=0.02, min = 0.005, max= 0.1)
fit_params3gx.add(name=('cen_g3'), expr='peak_split+cen_g2')
fit_params3gx.add('wid_g3', value=20, min = 5, max = 200)


amp_map1 = np.zeros_like(flux_map)
cen_map1= np.zeros_like(flux_map)
wid_map1 = np.zeros_like(flux_map)

amp_map2g_1 = np.zeros_like(flux_map)
cen_map2g_1 = np.zeros_like(flux_map)
wid_map2g_1 = np.zeros_like(flux_map)
amp_map2g_2 = np.zeros_like(flux_map)
cen_map2g_2 = np.zeros_like(flux_map)
wid_map2g_2 = np.zeros_like(flux_map)

amp_map3g_1 = np.zeros_like(flux_map)
cen_map3g_1 = np.zeros_like(flux_map)
wid_map3g_1 = np.zeros_like(flux_map)
amp_map3g_2 = np.zeros_like(flux_map)
cen_map3g_2 = np.zeros_like(flux_map)
wid_map3g_2 = np.zeros_like(flux_map)
amp_map3g_3 = np.zeros_like(flux_map)
cen_map3g_3 = np.zeros_like(flux_map)
wid_map3g_3 = np.zeros_like(flux_map)

amp_map1x = np.zeros_like(flux_map)
cen_map1x= np.zeros_like(flux_map)
wid_map1x = np.zeros_like(flux_map)

amp_map2g_1x = np.zeros_like(flux_map)
cen_map2g_1x = np.zeros_like(flux_map)
wid_map2g_1x = np.zeros_like(flux_map)
amp_map2g_2x = np.zeros_like(flux_map)
cen_map2g_2x = np.zeros_like(flux_map)
wid_map2g_2x = np.zeros_like(flux_map)

amp_map3g_1x = np.zeros_like(flux_map)
cen_map3g_1x = np.zeros_like(flux_map)
wid_map3g_1x = np.zeros_like(flux_map)
amp_map3g_2x = np.zeros_like(flux_map)
cen_map3g_2x = np.zeros_like(flux_map)
wid_map3g_2x = np.zeros_like(flux_map)
amp_map3g_3x = np.zeros_like(flux_map)
cen_map3g_3x = np.zeros_like(flux_map)
wid_map3g_3x = np.zeros_like(flux_map)

range1 = list(range(30,480))
range2 = list(range(104,345))

"""
PRIMO FIT A SPIRALE PER OTTENERE I PARAMETRI INIZIALI DA USARE IN UN SECONDO FIT
"""
for idxs in spiral_vect:
    prt = idxs.partition("-")
    ii = int(prt[2])    #x
    jj = int(prt[0])
    spec_tmp = datacube.data[:,jj,ii]
    spec_tmp = np.nan_to_num(spec_tmp)
    spec_tmp[0]=0
    if all(nlargest(2, spec_tmp) > 3.5*error):
        ##FIT CON 1 GAUSSIANA
        if jj > 1.14777 * ii +20:    #the pixel above the galaxy diagonal
            velmax = 90          #limit to select the blueshifted pixel
            velmin = -300
        else:
            velmax = 300
            velmin=-90
        p=1
        out = minimize(residual, fit_params1g, args=(x,p,), kws={'data': spec_tmp,'sigma': error/3})
        fit1 = residual(out.params, x, p)
        parvals = out.params.valuesdict()
        amplitude = parvals['amp_g1']
        stddev = parvals['wid_g1']
        mean = parvals['cen_g1']
        out_1g = [amplitude, mean, stddev]
        bic_1g = out.bic
        amp_map1[jj,ii], cen_map1[jj,ii], wid_map1[jj,ii] = out_1g
        fit_params1g.add('amp_g1', value=out_1g[0], min = 0.0025, max= 0.1)
        fit_params1g.add('cen_g1', value=out_1g[1], min = velmin, max= velmax)
        fit_params1g.add('wid_g1', value=out_1g[2], min = 1, max = 300)
        
        ##FIT CON 2 GAUSSIANE
        p = 2
        n = 2
        out = minimize(residual, fit_params2g, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/3})
        fit2 = residual(out.params, x, p)
        parvals = out.params.valuesdict()
        amplitude_2 = parvals['amp_g' + str(n)]
        stddev_2 = parvals['wid_g' + str(n)]
        mean_2 = parvals['cen_g' + str(n)]
        amplitude_1 = parvals['amp_g' + str(n-1)]
        stddev_1 = parvals['wid_g' + str(n-1)]
        mean_1 = parvals['cen_g' + str(n-1)]
        out_2g = [amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2]
        bic_2g = out.bic
        amp_map2g_1[jj,ii] , cen_map2g_1[jj,ii] , wid_map2g_1[jj,ii], amp_map2g_2[jj,ii], cen_map2g_2[jj,ii], wid_map2g_2[jj,ii] = out_2g
        fit_params2g.add('amp_g1', value=out_2g[0], min = 0.0025, max= 0.1)
        fit_params2g.add('cen_g1', value=out_2g[1], min = velmin, max= velmax)
        fit_params2g.add('wid_g1', value=out_2g[2], min = 1, max = 200)
        fit_params2g.add('amp_g2' , value=out_2g[3], min= 0.0025, max= 0.1)
        fit_params2g.add(name=('cen_g2'), expr='peak_split+cen_g1')
        fit_params2g.add('wid_g2', value=out_2g[5], min =1, max= 200)

        
        # ##FIT CON 3 GAUSSIANE
        p = 3
        n = 3
        out = minimize(residual, fit_params3g, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/3})
        fit3 = residual(out.params, x, p)
        parvals = out.params.valuesdict()
        amplitude_3 = parvals['amp_g' + str(n)]
        stddev_3 = parvals['wid_g' + str(n)]
        mean_3 = parvals['cen_g' + str(n)]
        amplitude_2 = parvals['amp_g' + str(n-1)]
        stddev_2 = parvals['wid_g' + str(n-1)]
        mean_2 = parvals['cen_g' + str(n-1)]
        amplitude_1 = parvals['amp_g' + str(n-2)]
        stddev_1 = parvals['wid_g' + str(n-2)]
        mean_1 = parvals['cen_g' + str(n-2)]
        bic_3g = out.bic
        out_3g = [amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2, amplitude_3, mean_3, stddev_3]
        amp_map3g_1[jj,ii] ,cen_map3g_1[jj,ii] ,wid_map3g_1[jj,ii] , amp_map3g_2[jj,ii] ,cen_map3g_2[jj,ii] ,wid_map3g_2[jj,ii] ,amp_map3g_3[jj,ii] , cen_map3g_3[jj,ii] , wid_map3g_3[jj,ii] = out_3g
        fit_params3g.add('amp_g1', value=out_3g[0], min = 0.0025, max= 0.1)
        fit_params3g.add('cen_g1', value=out_3g[1], min = velmin, max= velmax)
        fit_params3g.add('wid_g1', value=out_3g[2], min = 1, max = 200)
        fit_params3g.add('amp_g2' , value=out_3g[3], min= 0.0025, max= 0.1)
        fit_params3g.add(name=('cen_g2'), expr='peak_split+cen_g1')
        fit_params3g.add('wid_g2', value=out_3g[5], min =1, max= 200)
        fit_params3g.add('amp_g3' , value=out_3g[6], min= 0.0025, max= 0.1)
        fit_params3g.add(name=('cen_g3'), expr='peak_split+cen_g2')
        fit_params3g.add('wid_g3', value=out_3g[8], min =1, max= 200)

           
        ##FIT CON 1 GAUSSIANA con velocità iniziale data dal momento 1
        p=1
        fit_params1gx.add('cen_g1', value=M1[jj,ii], min = velmin, max= velmax)
        fit_params1gx.add('wid_g1', value=M2[jj,ii], min = 1, max = 300)
        out = minimize(residual, fit_params1gx, args=(x,p,), kws={'data': spec_tmp,'sigma': error/3})
        fit1x = residual(out.params, x, p)
        # mod1x[:,jj,ii] = fit1x
        parvals = out.params.valuesdict()
        amplitude = parvals['amp_g1']
        stddev = parvals['wid_g1']
        mean = parvals['cen_g1']
        out_1gx = [amplitude, mean, stddev]
        amp_map1x[jj,ii], cen_map1x[jj,ii], wid_map1x[jj,ii] = out_1gx
        bic_1gx = out.bic
        fit_params1gx.add('amp_g1', value=out_1gx[0], min = 0.0025, max= 0.1)

        
        ##FIT CON 2 GAUSSIANE con velocità iniziale data dal momento1
        p = 2
        n = 2
        fit_params2gx.add('cen_g1', value=M1[jj,ii], min = velmin, max= velmax)
        out = minimize(residual, fit_params2gx, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/3})
        fit2x = residual(out.params, x, p)
        # mod2x[:,jj,ii] = fit2x
        parvals = out.params.valuesdict()
        amplitude_2 = parvals['amp_g' + str(n)]
        stddev_2 = parvals['wid_g' + str(n)]
        mean_2 = parvals['cen_g' + str(n)]
        amplitude_1 = parvals['amp_g' + str(n-1)]
        stddev_1 = parvals['wid_g' + str(n-1)]
        mean_1 = parvals['cen_g' + str(n-1)]
        out_2gx = [amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2]
        amp_map2g_1x[jj,ii] , cen_map2g_1x[jj,ii] , wid_map2g_1x[jj,ii], amp_map2g_2x[jj,ii], cen_map2g_2x[jj,ii], wid_map2g_2x[jj,ii] = out_2gx
        bic_2gx = out.bic
        fit_params2gx.add('amp_g1', value=out_2gx[0], min = 0.0025, max= 0.1)
        fit_params2gx.add('wid_g1', value=out_2gx[2], min = 1, max = 200)
        fit_params2gx.add('amp_g2' , value=out_2gx[3], min= 0.0025, max= 0.1)
        fit_params2gx.add(name=('cen_g2'), expr='peak_split+cen_g1')
        fit_params2gx.add('wid_g2', value=out_2gx[5], min =1, max= 200)

        # # ##FIT CON 3 GAUSSIANE con velocità iniziale data dal moment1
        p = 3
        n = 3
        fit_params3gx.add('cen_g1', value=M1[jj,ii], min = velmin, max= velmax)
        out = minimize(residual, fit_params3gx, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/3})
        fit3x = residual(out.params, x, p)
        # mod3[:,jj,ii] = fit3x
        parvals = out.params.valuesdict()
        amplitude_3 = parvals['amp_g' + str(n)]
        stddev_3 = parvals['wid_g' + str(n)]
        mean_3 = parvals['cen_g' + str(n)]
        amplitude_2 = parvals['amp_g' + str(n-1)]
        stddev_2 = parvals['wid_g' + str(n-1)]
        mean_2 = parvals['cen_g' + str(n-1)]
        amplitude_1 = parvals['amp_g' + str(n-2)]
        stddev_1 = parvals['wid_g' + str(n-2)]
        mean_1 = parvals['cen_g' + str(n-2)]
        out_3gx = [amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2, amplitude_3, mean_3, stddev_3]
        bic_3gx = out.bic
        fit_params3gx.add('amp_g1', value=out_3gx[0], min = 0.0025, max= 0.1)
        fit_params3gx.add('cen_g1', value=out_3gx[1], min = velmin, max= velmax)
        fit_params3gx.add('wid_g1', value=out_3gx[2], min = 1, max = 200)
        fit_params3gx.add('amp_g2' , value=out_3gx[3], min= 0.0025, max= 0.1)
        fit_params3gx.add(name=('cen_g2'), expr='peak_split+cen_g1')
        fit_params3gx.add('wid_g2', value=out_3gx[5], min =1, max= 200)
        fit_params3gx.add('amp_g3' , value=out_3gx[6], min= 0.0025, max= 0.1)
        fit_params3gx.add(name=('cen_g3'), expr='peak_split+cen_g2')
        fit_params3gx.add('wid_g3', value=out_3gx[8], min =1, max= 200)
        
        
        if jj in range1 and ii in range2:
            if bic_1g > bic_1gx:    
                amp_map1[jj,ii] = amp_map1x[jj,ii]
                cen_map1[jj,ii] = cen_map1x[jj,ii]
                wid_map1[jj,ii] = wid_map1x[jj,ii]
            if bic_2g > bic_2gx:
                amp_map2g_1[jj,ii] = amp_map2g_1x[jj,ii]
                cen_map2g_1[jj,ii] = cen_map2g_1x[jj,ii]
                wid_map2g_1[jj,ii] = wid_map2g_1x[jj,ii]
                amp_map2g_2[jj,ii] = amp_map2g_2x[jj,ii]
                cen_map2g_2[jj,ii] = cen_map2g_2x[jj,ii]
                wid_map2g_2[jj,ii] = wid_map2g_2x[jj,ii]
            if bic_3g > bic_3gx:
                amp_map3g_1[jj,ii] = amp_map3g_1x[jj,ii]
                cen_map3g_1[jj,ii] = cen_map3g_1x[jj,ii]
                wid_map3g_1[jj,ii] = wid_map3g_1x[jj,ii]
                amp_map3g_2[jj,ii] = amp_map3g_2x[jj,ii]
                cen_map3g_2[jj,ii] = cen_map3g_2x[jj,ii]
                wid_map3g_2[jj,ii] = wid_map3g_2x[jj,ii]
                amp_map3g_3[jj,ii] = amp_map3g_3x[jj,ii]
                cen_map3g_3[jj,ii] = cen_map3g_3x[jj,ii]
                wid_map3g_3[jj,ii] = wid_map3g_3x[jj,ii]



"""
FIT NORMALE USANDO COME PARAMETRI INIZIALI I BEST PARAMS DEL FIT A SPIRALE
"""
for jj in range(Ny):
    for ii in range(Nx):
        spec_tmp = datacube.data[:,jj,ii]
        spec_tmp = np.nan_to_num(spec_tmp)
        spec_tmp[0]=0
        if all(nlargest(2, spec_tmp) > 3.5*error):
            ##FIT CON 1 GAUSSIANA
            if jj > 1.14777 * ii +20:    #the pixel above the galaxy diagonal
                velmax = 90          #limit to select the blueshifted pixel
                velmin = -300
            else:
                velmax = 300
                velmin=-90
            p=1
            fit_params1g.add('amp_g1', value=amp_map1[jj,ii], min = 0.0025, max= 0.1)
            fit_params1g.add('cen_g1', value=cen_map1[jj,ii], min = velmin, max= velmax)
            fit_params1g.add('wid_g1', value=wid_map1[jj,ii], min = 1, max = 300)
            out = minimize(residual, fit_params1g, args=(x,p,), kws={'data': spec_tmp,'sigma': error/3})
            fit1 = residual(out.params, x, p)
            bic_1g = out.bic
            
            ##FIT CON 2 GAUSSIANE
            p = 2
            n = 2
            fit_params2g.add('amp_g1', value=amp_map2g_1[jj,ii], min = 0.0025, max= 0.1)
            fit_params2g.add('cen_g1', value=cen_map2g_1[jj,ii], min = velmin, max= velmax)
            fit_params2g.add('wid_g1', value=wid_map2g_1[jj,ii], min = 1, max = 200)
            fit_params2g.add('amp_g2' , value=amp_map2g_2[jj,ii], min= 0.0025, max= 0.1)
            fit_params2g.add('cen_g2', value = cen_map2g_2[jj,ii], min = velmin, max = velmax)
            fit_params2g.add('wid_g2', value=wid_map2g_2[jj,ii], min =1, max= 200)
            out = minimize(residual, fit_params2g, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/3})
            fit2 = residual(out.params, x, p)
            bic_2g = out.bic

            
            # ##FIT CON 3 GAUSSIANE
            p = 3
            n = 3
            fit_params3g.add('amp_g1', value=amp_map3g_1[jj,ii], min = 0.0025, max= 0.1)
            fit_params3g.add('cen_g1', value=cen_map3g_1[jj,ii], min = velmin, max= velmax)
            fit_params3g.add('wid_g1', value=wid_map3g_1[jj,ii], min = 1, max = 200)
            fit_params3g.add('amp_g2' , value=amp_map3g_2[jj,ii], min= 0.0025, max= 0.1)
            fit_params3g.add('cen_g2', value=cen_map3g_2[jj,ii], min = velmin, max= velmax)
            fit_params3g.add('wid_g2', value=wid_map3g_2[jj,ii], min =1, max= 200)
            fit_params3g.add('amp_g3' , value=amp_map3g_3[jj,ii], min= 0.0025, max= 0.1)
            fit_params3g.add('cen_g3', value = cen_map3g_3[jj,ii], min = velmin, max = velmax)
            fit_params3g.add('wid_g3', value=wid_map3g_3[jj,ii], min =1, max= 200)
            out = minimize(residual, fit_params3g, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/3})
            fit3 = residual(out.params, x, p)
            bic_3g = out.bic

            if jj in range1 and ii in range2:
                if bic_1g < bic_2g and bic_1g < bic_3g  and bic_2g - bic_1g > 2.3 and bic_3g - bic_1g > 2.3:
                    flux_map[jj,ii] = np.nansum(fit1) * dv
                    vel_map[jj,ii] = np.nansum((fit1*velocity)) * dv / flux_map[jj,ii]
                    vdisp_map[jj,ii] = np.nansum((fit1*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
                    vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                elif bic_2g < bic_1g and bic_2g < bic_3g and bic_1g - bic_2g > 2.3 and bic_3g - bic_2g > 2.3: #and bic_2g < bic_3g: # and bic_1g - bic_2g > 2.3
                    flux_map[jj,ii] = np.nansum(fit2) * dv
                    vel_map[jj,ii] = np.nansum((fit2*velocity)) * dv/flux_map[jj,ii]
                    vdisp_map[jj,ii] = np.nansum((fit2*(velocity-vel_map[jj,ii])**2)) * dv /flux_map[jj,ii]  
                    vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                elif bic_3g < bic_1g and bic_3g < bic_2g and bic_1g - bic_3g > 2.3 and bic_2g - bic_3g > 2.3:
                    flux_map[jj,ii] = np.nansum(fit3) * dv
                    vel_map[jj,ii] = np.nansum((fit3*velocity)) * dv / flux_map[jj,ii]
                    vdisp_map[jj,ii] = np.nansum((fit3*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
                    vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                elif bic_2g < bic_1g and bic_2g < bic_3g and bic_1g - bic_2g < 2.3:
                    flux_map[jj,ii] = np.nansum(fit1) * dv 
                    vel_map[jj,ii] = np.nansum((fit1*velocity)) * dv / flux_map[jj,ii]
                    vdisp_map[jj,ii] = np.nansum((fit1*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]
                    vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                elif bic_3g < bic_1g and bic_3g < bic_2g and bic_2g - bic_3g < 2.3:
                    flux_map[jj,ii] = np.nansum(fit2) * dv
                    vel_map[jj,ii] = np.nansum((fit2*velocity)) * dv / flux_map[jj,ii]
                    vdisp_map[jj,ii] = np.nansum((fit2*(velocity-vel_map[jj,ii])**2)) * dv / flux_map[jj,ii]  
                    vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5



plt.figure(figsize = (12,4))

plt.subplot(131)
plt.imshow(flux_map, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7)
plt.subplot(132)
plt.imshow(vel_map, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar(shrink = 0.7)
plt.subplot(133)
plt.imshow(vdisp_map, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar(shrink = 0.7)


# hdu = fits.PrimaryHDU(res1)
# hdul = fits.HDUList([hdu])
# hdul.writeto('res1g.fits')
# hdu = fits.PrimaryHDU(res2)
# hdul = fits.HDUList([hdu])
# hdul.writeto('res2g.fits')
# hdu = fits.PrimaryHDU(res3)
# hdul = fits.HDUList([hdu])
# hdul.writeto('res3g.fits')
# hdu = fits.PrimaryHDU(mod1)
# hdul = fits.HDUList([hdu])
# hdul.writeto('mod1.fits')
# hdu = fits.PrimaryHDU(mod2)
# hdul = fits.HDUList([hdu])
# hdul.writeto('mod2g.fits')
# hdu = fits.PrimaryHDU(mod3)
# hdul = fits.HDUList([hdu])
# hdul.writeto('mod3g.fits')

# hdu = fits.PrimaryHDU(mod)
# hdul = fits.HDUList([hdu])
# hdul.writeto('model_double_spiral.fits')
# # hdu = fits.PrimaryHDU(res)
# # hdul = fits.HDUList([hdu])
# # hdul.writeto('residual.fits')



# hdu = fits.PrimaryHDU(flux_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('flux_map_double_spiral.fits')
# hdu = fits.PrimaryHDU(vel_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('vel_map_double_spiral.fits')
# hdu = fits.PrimaryHDU(vdisp_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('vdisp_map_double_spiral.fits')