#import python libraries
import numpy as np 
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.wcs as wcs
import os 
import time
#from astropy.nddata import Cutout2D
import astropy.constants as K
import astropy.units as u
from astropy.cosmology import Planck15 as p15
import scipy.ndimage
from lmfit import minimize, Parameters, report_fit, Parameter
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
# report_fit(out, modelpars=p_true)
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

    
#Generating moments map
#datacube.data = np.where(datacube.data<0, 0*datacube.data, datacube.data)
#mask_cube = datacube.data>3*error

#generete velocity maps 
mask_cube = np.where(datacube.data > 3*error, datacube.data, np.nan)
moment0 = np.nansum(mask_cube, axis = (0))*dv
#moment0 = np.nan_to_num(moment0)
#moment1 = np.full_like(moment0,np.nan)
#vdisp_map2 = np.full_like(moment0,np.nan)
#moment2 = np.full_like(moment0,np.nan)

moment1 = np.nansum(datacube.data[:,:,:]*velocity[:,np.newaxis,np.newaxis], axis=0)*dv / moment0
# avoid division by 0 or neg values in sqrt
thr = np.nanpercentile(moment0[np.where(moment0>0)],0.01)
moment0[np.where(moment0<thr)]=np.nan
moment2 = np.sqrt(np.nansum(np.power(datacube.data[:, :, :] * (velocity[:, np.newaxis, np.newaxis] - moment1[np.newaxis, :, :]),2), axis=0) * dv / moment0 )

plt.figure(figsize = (12,4))

plt.subplot(131)
plt.imshow(moment0, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7)
plt.subplot(132)
plt.imshow(moment1, origin = 'lower', vmin = -300, vmax = 300, cmap ='jet')
plt.colorbar(shrink = 0.7)
plt.subplot(133)
plt.imshow(moment2, origin = 'lower',  vmin = 0, vmax =200, cmap = 'jet')
plt.colorbar(shrink = 0.7)
        

# for ii in range(Nx):
#     for jj in range(Ny):
#         if moment0[jj,ii] > 3*error:
#             moment1[jj,ii] = np.nansum((datacube.data[:,jj,ii]*velocity))/moment0[jj,ii]
#             vdisp_map2[jj,ii] = np.nansum((datacube.data[:,jj,ii]*(velocity-moment1[jj,ii])**2))/moment0[jj,ii]
#             moment2[jj,ii] = vdisp_map2[jj,ii]**0.5
  

#moment1[moment0 < 3*error] = np.nan
#moment2[moment0 < 3*error] = np.nan

                    

#Creating the matrix-coordinates that would be spiralized
mat_tmp =  np.empty([Nx, Ny], dtype='<U7')
for i in range(Nx):
    for j in range(Ny):
        mat_tmp[i,j] = (str(j) + '-' + str(i))   #controllare indici matrice/mappe momenti
        
#Spiralizing the matrix
spiral_vect = spiral_mat_to_vect(mat_tmp)
spiral_vect = invers_spiral(spiral_vect)

flux_map_tmp = np.nansum(datacube.data, axis = (0))
flux_map = np.full_like(flux_map_tmp, np.nan)
vel_map = np.full_like(flux_map, np.nan)
vdisp_map = np.full_like(flux_map, np.nan)


for idxs in spiral_vect[:30]:
    prt = idxs.partition("-")
    ii = int(prt[2])    #x
    jj = int(prt[0])    #y
    spec_tmp = datacube.data[:,jj,ii]
    spec_tmp = np.nan_to_num(spec_tmp)
    if moment0[jj,ii] > 3*error: #all(nlargest(2, spec_tmp) > 3*error) and np.isnan(moment0[jj,ii]) == False and np.isnan(moment1[jj,ii]) == False and np.isnan(moment2[jj,ii]) == False:
        ##FIT CON 1 GAUSSIANA
        p=1
        fit_params1g = Parameters()
        fit_params1g.add('amp_g1', value=moment0[jj,ii]/56, min = 0.0025, max= 0.1)
        fit_params1g.add('cen_g1', value=moment1[jj,ii], min = -300, max= 300)
        fit_params1g.add('wid_g1', value=moment2[jj,ii], min = 10, max = 200)
        out1g = minimize(residual, fit_params1g, args=(x,p,), kws={'data': spec_tmp,'sigma': error/3})
        fit1 = residual(out1g.params, x, p)        
        bic_1g = out1g.bic
        # print('##')
        # print('1gaussian_fit pixel:' + str(ii) + '-' + str(jj))
        # # report_fit(out, modelpars=p_true)
        # report_fit(out1g)
        plt.figure(figsize = (12,4))
        plt.plot(x, spec_tmp, label='data')
        plt.plot(x, data*0,':',color = 'black')
        plt.plot(x, fit1, label='best fit')
        plt.xlabel('velocity [km/s]')
        plt.ylabel('flux [mJy]')
        plt.title('1Gaussian_model_pixel' + str(ii) + '-' + str(jj) )
        plt.legend()
        plt.show()
        ##FIT CON 2 GAUSSIANE
        p = 2
        n = 2
        fit_params2g = Parameters()
        fit_params2g.add('amp_g1', moment0[jj,ii]/56, min = 0.0025, max= 0.1)
        fit_params2g.add('cen_g1', moment1[jj,ii], min = -300, max= 300)
        fit_params2g.add('wid_g1', value= moment2[jj,ii], min = 10, max = 200)
        fit_params2g.add('amp_g2' , value=moment0[jj,ii]/56, min= 0.0025, max= 0.1)
        fit_params2g.add(name='peak_split', value=0, min=-300, max=300, vary=True)
        fit_params2g.add(name=('cen_g2'), expr='peak_split+cen_g1')
        fit_params2g.add('wid_g2', value=moment2[jj,ii], min =10, max= 200)
        out2g = minimize(residual, fit_params2g, args=(x,p,), kws={'data': spec_tmp, 'sigma':error/3})
        fit2 = residual(out2g.params, x, p)
        bic_2g = out2g.bic
        # print('##')
        # print(str(n) + 'gaussianfit' + 'pixel:' + str(ii) + '-' + str(jj))
        # # report_fit(out, modelpars=p_true)
        # report_fit(out2g)
        plt.figure(figsize = (12,4))
        plt.plot(x, spec_tmp, label='data')
        plt.plot(x, data*0,':',color = 'black')
        plt.plot(x, fit2, label='best fit')
        plt.xlabel('velocity [km/s]')
        plt.ylabel('flux [mJy]')
        plt.title('2Gaussian_model_pixel' + str(ii) + '-' + str(jj) )
        plt.legend()
        plt.show()
        
        # ##FIT CON 3 GAUSSIANE
        # p = 3
        # n = 3
        # out = minimize(residual, fit_params3g, args=(x,p,), kws={'data': spec_tmp, 'sigma':error})
        # fit3 = residual(out.params, x, p)
        # parvals = out.params.valuesdict()
        # amplitude_3 = parvals['amp_g' + str(n)]
        # stddev_3 = parvals['wid_g' + str(n)]
        # mean_3 = parvals['cen_g' + str(n)]
        # amplitude_2 = parvals['amp_g' + str(n-1)]
        # stddev_2 = parvals['wid_g' + str(n-1)]
        # mean_2 = parvals['cen_g' + str(n-1)]
        # amplitude_1 = parvals['amp_g' + str(n-2)]
        # stddev_1 = parvals['wid_g' + str(n-2)]
        # mean_1 = parvals['cen_g' + str(n-2)]
        # out_3g = [amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2, amplitude_3, mean_3, stddev_3]
        # bic_3g = out.bic
        # fit_params3g.add('amp_g1', value=out_3g[0], min = 0.0025, max= 0.1)
        # fit_params3g.add('cen_g1', value=out_3g[1], min = -300, max= 300)
        # fit_params3g.add('wid_g1', value=out_3g[2], min = 10, max = 300)
        # fit_params3g.add('amp_g2' , value=out_3g[3], min= 0.0025, max= 0.1)
        # fit_params3g.add(name=('cen_g2'), expr='peak_split+cen_g1')
        # fit_params3g.add('wid_g2', value=out_3g[5], min =10, max= 300)
        # fit_params3g.add('amp_g3' , value=out_3g[6], min= 0.0025, max= 0.1)
        # fit_params3g.add(name=('cen_g3'), expr='peak_split+cen_g2')
        # fit_params3g.add('wid_g3', value=out_3g[8], min =10, max= 300)
        # plt.figure(figsize = (12,4))
        # plt.plot(x, spec_tmp, label='data')
        # plt.plot(x, data*0,':',color = 'black')
        # plt.plot(x, fit3, label='best fit')
        # plt.xlabel('velocity [km/s]')
        # plt.ylabel('flux [mJy]')
        # plt.title('3Gaussian_model_pixel' + str(ii) + '-' + str(jj) )
        # plt.legend()
        # plt.show()
        if jj in range (30,480) and ii in range(104,345):
            if bic_1g < bic_2g and bic_2g - bic_1g > 2.3: #and bic_1g < bic_3g:  # and bic_2g - bic_1g > 2.3 and bic_3g - bic_1g > 2.3
                flux_map[jj,ii] = np.nansum(fit1) * dv
                vel_map[jj,ii] = np.nansum((fit1*velocity))* dv/flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit1*(velocity-vel_map[jj,ii])**2))* dv/flux_map[jj,ii]
                vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                # print(flux_map[jj,ii], vel_map[jj,ii,], vdisp_map[jj,ii], '1g')
    
                # print('pixel:' + str(ii) + '-' + str(jj))
                # print(bic_1g, bic_2g, bic_3g, 'bestfit= 1G')
            elif bic_2g < bic_1g and bic_1g - bic_2g > 2.3: #and bic_2g < bic_3g: #and bic_3g - bic_2g > 2.3 and bic_1g - bic_2g > 2.3
                flux_map[jj,ii] = np.nansum(fit2) * dv
                vel_map[jj,ii] = np.nansum((fit2*velocity))* dv/flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit2*(velocity-vel_map[jj,ii])**2))* dv/flux_map  [jj,ii]
                vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                # print(flux_map[jj,ii], vel_map[jj,ii,], vdisp_map[jj,ii], '2g')
                # print('pixel:' + str(ii) + '-' + str(jj))
                # print(bic_1g, bic_2g, bic_3g, 'bestfit= 2G')
            # elif bic_3g < bic_1g and bic_3g < bic_2g: #and bic_1g - bic_3g > 2.3 and bic_2g - bic_3g > 2.3
            #     flux_map[jj,ii] = (out_3g[0] + out_3g[3] + out_3g[6])/3
            #     vel_map[jj,ii] = np.nansum((fit3*velocity))/flux_map[jj,ii]
            #     vdisp_map[jj,ii] = np.nansum((fit3*(velocity-vel_map[jj,ii])**2))/flux_map[jj,ii]
            #     vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
            #     # print(flux_map[jj,ii], vel_map[jj,ii,], vdisp_map[jj,ii], '2g')
            #     # print('pixel:' + str(ii) + '-' + str(jj))
            #     # print(bic_1g, bic_2g, bic_3g, 'bestfit= 3G')
            else:
                flux_map[jj,ii] = np.nansum(fit1) * dv
                vel_map[jj,ii] = np.nansum((fit1*velocity)) *dv /flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit1*(velocity-vel_map[jj,ii])**2))*dv/flux_map[jj,ii]
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


# hdu = fits.PrimaryHDU(flux_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('flux_map_spiral_13_2g_init_moments.fits')
# hdu = fits.PrimaryHDU(vel_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('vel_map_spiral_13_2g_init_moments.fits')
# hdu = fits.PrimaryHDU(vdisp_map)
# hdul = fits.HDUList([hdu])
# hdul.writeto('vdisp_map_spiral_13_2g_init_moments.fits')



