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
        model = pars['amp_g1'] * np.exp(-argu1) + pars['C_g1']
    if p == 2:
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2'] * np.exp(-argu2)) + pars['C_g2']
        modelz = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2'] * np.exp(-argu2))
        model_O = pars['amp_g1'] * np.exp(-argu1)
    if p == 3:
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3'] * np.exp(-argu3)) + pars['C_g3']
        modelz = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3'] * np.exp(-argu3))
        model_O = pars['amp_g1'] * np.exp(-argu1) 
    if p == 4:
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        argu4 = (x - pars['cen_g4'])**2 / (2*(pars['wid_g4'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3']*np.exp(-argu3) + pars['amp_g4']*np.exp(-argu4)) + pars['C_g4']
    if p == 5:
        argu2 = (x - pars['cen_g2'])**2 / (2*(pars['wid_g2'])**2)
        argu3 = (x - pars['cen_g3'])**2 / (2*(pars['wid_g3'])**2)
        argu4 = (x - pars['cen_g4'])**2 / (2*(pars['wid_g4'])**2)
        argu5 = (x - pars['cen_g5'])**2 / (2*(pars['wid_g5'])**2)
        model = (pars['amp_g1'] * np.exp(-argu1) + pars['amp_g2']*np.exp(-argu2) + pars['amp_g3']*np.exp(-argu3) + pars['amp_g4']*np.exp(-argu4) + pars['amp_g5']*np.exp(-argu5)) + pars['C_g5']
    
    if data is None:
            return model, modelz, model_O
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
wavelenght_alpha = wavelenght_m[1340:1520]

frequency = K.c.to('m s-1') / wavelenght_m #Hz
frequency = 1 * frequency.to('THz')

frequency_beta = frequency[8:98]
frequency_O = frequency[110:210]
frequency_alpha = frequency[1340:1520]




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
error_alpha = np.std(noise[1340:1520, :, :])
error_tot = np.std(noise[:, :, :])

print("rms  = {:2f} mJy".format(error_O))
print("####################")

#----------------------------------------------------------------------------

# Fit and plot of the total spectrum
# First Line: O_III 4862
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
fit, fitz , fit0 = residual(out.params, x, p)
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

del(x)
#----------------------------------------------------------------------------

# Making 3 different subcubes for the 3 part of the fit (H_beta, O_III doublet, H_alpha e NII doublet)

#H_beta

O_III = datacube.data[8:98, :, :]
w_beta = wavelenght[8:98]
O_III = datacube.data[110:210, :, :]
w_O = wavelenght[110:210]
H_alpha = datacube.data[1340:1520, :, :]
w_alpha = wavelenght[1340:1520]

#----------------------------------------------------------------------------


# Generating moments map without fitting (BRUTE MOMENTS MAP)
# H_beta

#datacube.data = np.where(datacube.data<0, 0*datacube.data, datacube.data)
mask_cube = np.where(O_III > 2.5*error_O, O_III, np.nan)
M0 = np.nansum(mask_cube, axis = (0)) * dlambda
M1 = np.nansum(mask_cube[:,:,:]*velocity_O[:,np.newaxis,np.newaxis], axis=0) *dlambda / M0
# avoid division by 0 or neg values in sqrt
# thr = np.nanpercentile(M0[np.where(M0>0)],0.01)
thr = 2.5*error_O
M0[np.where(M0<thr)]=np.nan
M2 = np.sqrt(np.nansum(np.power(O_III[:, :, :] * (velocity_O[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :]),2), axis=0) *dlambda / M0 )     


plt.figure(figsize = (12,4))

plt.subplot(131)
plt.imshow(M0, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7)
plt.title('Flux')
plt.subplot(132)
plt.imshow(M1, origin = 'lower',vmin= -1000, vmax= 1000, cmap ='jet')
plt.colorbar(shrink = 0.7)
plt.title('Velocity')
plt.subplot(133)
plt.imshow(M2, origin = 'lower',  vmin = 0, vmax= 2000, cmap = 'jet')
plt.colorbar(shrink = 0.7)
plt.title('Velocity Dispersion')
plt.suptitle('brute moment maps for O_III doublet')

plt.show()

#----------------------------------------------------------------------------

# generete model maps 

mod = np.zeros_like(O_III)
mod1 = np.zeros_like(O_III)
# mod2 = np.zeros_like(datacube.data)
# modx = np.zeros_like(datacube.data)
mod1x = np.zeros_like(O_III)
# mod2x= np.zeros_like(datacube.data)
# # res = np.zeros_like(datacube.data)
# # res1 = np.zeros_like(datacube.data)
# # res2 = np.zeros_like(datacube.data)
# # res3 = np.zeros_like(datacube.data)

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
flux_map_tmp = np.nansum(O_III, axis = (0))
flux_map = np.full_like(flux_map_tmp, np.nan)
flux_map_O = np.full_like(flux_map_tmp, np.nan)
vel_map = np.full_like(flux_map, np.nan)
vdisp_map = np.full_like(flux_map, np.nan)
# amp1_1err= np.full_like(flux_map, np.nan)
# cen1_1err= np.full_like(amp1_1err, np.nan)
# wid1_1err= np.full_like(amp1_1err, np.nan)
# amp2_1err =np.full_like(amp1_1err, np.nan)
# amp2_2err =np.full_like(amp1_1err, np.nan)
# cen2_1err =np.full_like(amp1_1err, np.nan)
# cen2_2err =np.full_like(amp1_1err, np.nan)
# wid2_1err =np.full_like(amp1_1err, np.nan)
# wid2_2err=np.full_like(amp1_1err, np.nan)
# amp1x_1err=np.full_like(amp1_1err, np.nan)
# cen1x_1err=np.full_like(amp1_1err, np.nan)
# wid1x_1err=np.full_like(amp1_1err, np.nan)
# amp2x_1err=np.full_like(amp1_1err, np.nan)
# amp2x_2err=np.full_like(amp1_1err, np.nan)
# cen2x_1err=np.full_like(amp1_1err, np.nan)
# cen2x_2err=np.full_like(amp1_1err, np.nan)
# wid2x_1err=np.full_like(amp1_1err, np.nan)
# wid2x_2err=np.full_like(amp1_1err, np.nan)
# amperr = np.full_like(amp1_1err, np.nan)
# velerr = np.full_like(amp1_1err, np.nan)
# disperr = np.full_like(amp1_1err, np.nan) 
# #velarr = np.full_like(amp1_1err, np.nan) 
# #disparr = np.full_like(amp1_1err, np.nan) 

# #----------------------------------------------------------------------------

#Initialize the initial parameters for the lmfit of the first pixel (the center)
fit_params2g = Parameters()
fit_params2g.add('amp_g1', value=300, min =20, max = 1000)
fit_params2g.add('cen_g1', value=0,)
fit_params2g.add('wid_g1', value=100, min = 10, max = 500)
fit_params2g.add(name='peak_split', value=3000, min=500, max=3500, vary=True)
fit_params2g.add('C_g2', value = 5, min = 0 )
fit_params2g.add('amp_g2', expr = 'amp_g1 / 3', min =20, max = 300)
fit_params2g.add('cen_g2', expr = 'cen_g1 - peak_split', min = -3500, max= -2000)
fit_params2g.add('wid_g2', value=100, min = 10, max = 500)


fit_params2gx = Parameters()
fit_params2gx.add('amp_g1', value=300, min = 20, max = 1000)
fit_params2gx.add('cen_g1', value=0)
fit_params2gx.add('wid_g1', value=150, min = 10, max = 500)
fit_params2gx.add('C_g2', value = 5, min = 0 )
fit_params2gx.add('amp_g2', expr = 'amp_g1 / 3', min =20, max = 300)
fit_params2gx.add('cen_g2', value = -3000, min = -4000, max= -2000)
fit_params2gx.add('wid_g2', expr = 'wid_g1', min = 10, max = 500)


# #----------------------------------------------------------------------------

# # velmin, velmax = -300, 300
range1 = list(range(0,320))        # y range for the final maps  
range2 = list(range(0,318))       # x range for the final maps
printplot = False
printfit = False
    

#----------------------------------------------------------------------------
x = velocity_O
p=2
# Spiral Fit
for idxs in spiral_vect:
    prt = idxs.partition("-")                       # Take the coordinates of the pixel in the spiral fitting
    ii = int(prt[2])    #x
    jj = int(prt[0])    #y
    spec_tmp = O_III[:,jj,ii]
    spec_tmp = np.nan_to_num(spec_tmp)
    # spec_tmp[0]=0
    if all(nlargest(2, spec_tmp) > 1.5*error_O):      # thershold on the 2 biggest values of the spectra
    # if flux_map_tmp[jj,ii]>4.2*error:
    #     spec_tmp = datacube.data[:,jj,ii]
    #     spec_tmp = np.nan_to_num(spec_tmp)
    #     spec_tmp[0]=0
    # if flux_map_tmp[jj,ii] > 2.5 * error_beta:  
        
        # FIT WITH 2 GAUSSIAN
        if jj > ii :                  # imposing differnt limit on the pixel above the galaxy diagonal
            velmax = 150                             # limit to select the blueshifted pixel
            velmin = -600
        else:
            velmax = 600
            velmin=-150
      
        # fit_params1g.add('cen_g1', value=M1[jj,ii], min = velmin, max= velmax)
        out = minimize(residual, fit_params2g, args=(x,p,), kws={'data': spec_tmp,'sigma': error_O * 2})
        fit1,fit1z, fit_O = residual(out.params, x, p)
        mod1[:,jj,ii] = fit1
        parvals = out.params.valuesdict()

        amplitude_1 = parvals['amp_g1']
        stddev_1 = parvals['wid_g1']
        mean_1 = parvals['cen_g1']
        split = parvals['peak_split']
        cost = parvals['C_g2']
        amplitude_2 = parvals['amp_g2']
        stddev_2 = parvals['wid_g2']
        mean_2 = parvals['cen_g2' ]
        # amp1_1err[jj,ii] = out.params['amp_g1'].stderr
        # cen1_1err[jj,ii]= out.params['cen_g1'].stderr
        # wid1_1err[jj,ii] = out.params['wid_g1'].stderr
        out_1g = [amplitude_1, mean_1, stddev_1, split, cost, amplitude_2, mean_2, stddev_2]
        bic_1g = out.bic
        # chi1 = out.redchi
        
        # Pass the best fit parameters as initial parameters for the next pixel to fit
        fit_params2g.add('amp_g1', value=out_1g[0], min = 30, max = 1000)
        fit_params2g.add('cen_g1', value=out_1g[1], min= velmin, max=velmax)
        fit_params2g.add('wid_g1', value=out_1g[2], min = 10, max=500)
        fit_params2g.add('peak_split', value = out_1g[3], min=500, max=3500, vary=True)
        fit_params2g.add('C_g2', value = out_1g[4], min= 0)
        fit_params2g.add('amp_g2' , expr = 'amp_g1 / 3', min= 10, max= 300)
        fit_params2g.add(name=('cen_g2'), expr='cen_g1 - peak_split')
        fit_params2g.add('wid_g2', value=out_1g[7], min =10, max= 500)
        
        
        if printplot == True:
            plt.figure(figsize = (12,4))
            plt.plot(x, spec_tmp, label='data')
            plt.plot(x, x*0,':',color = 'black')
            plt.plot(x, fit1, label='best fit')
            plt.xlabel('velocity [km/s]')
            plt.ylabel('flux [Jy]')
            plt.title('2Gaussian_model_pixel' + str(ii) + '-' + str(jj) )
            plt.legend()
            plt.show()
        if printfit == True:
            print('##')
            print('2gaussian_fit pixel:' + str(ii) + '-' + str(jj))
            report_fit(out)
        
        ##FIT WITH 2 GAUSSIAN with initial velocity taken from the "brute" moment 1
        # fit_params1gx.add('wid_g1', value=M2[jj,ii], min = 10, max=900)
        out = minimize(residual, fit_params2gx, args=(x,p,), kws={'data': spec_tmp,'sigma': error_O * 2})
        fit1x, fit1xz , fit_Ox= residual(out.params, x, p)
        mod1x[:,jj,ii] = fit1x
        parvals = out.params.valuesdict()
        amplitude_1 = parvals['amp_g1']
        stddev_1 = parvals['wid_g1']
        mean_1 = parvals['cen_g1']
        cost = parvals['C_g2']
        amplitude_2 = parvals['amp_g2']
        stddev_2 = parvals['wid_g2']
        mean_2 = parvals['cen_g2' ]
        # amp1x_1err[jj,ii] = out.params['amp_g1'].stderr
        # cen1x_1err[jj,ii]= out.params['cen_g1'].stderr
        # wid1x_1err[jj,ii] = out.params['wid_g1'].stderr
        out_1gx = [amplitude_1, mean_1, stddev_1, cost, amplitude_2, mean_2, stddev_2]
        bic_1gx = out.bic
        # chi1x = out.redchi
        
        # Pass the best fit parameters as initial parameters for the next pixel to fit
        # fit_params1gx.add('amp_g1', value=out_1gx[0], min = 100)
        # fit_params1gx.add('cen_g1', value=out_1g[1],)
        fit_params2gx.add('amp_g1', value=out_1gx[0], min = 10, max = 1000)
        fit_params2gx.add('cen_g1', value=out_1gx[1], min = velmin, max= velmax)
        fit_params2gx.add('wid_g1', value=out_1gx[2], min = 10, max=500)
        fit_params2gx.add('C_g2', value = out_1gx[3], min= 0)
        fit_params2gx.add('amp_g2' , expr = 'amp_g1 / 3', min= 10, max= 300)
        fit_params2gx.add(name=('cen_g2'), value = out_1gx[5])
        fit_params2gx.add('wid_g2', expr = 'wid_g1', min =10, max= 500)
        
        if printplot == True:
            # # res1[:,jj,ii] = residual(out.params, x, p, data, error/3) 
            plt.figure(figsize = (12,4))
            plt.plot(x, spec_tmp, label='data')
            plt.plot(x, x*0,':',color = 'black')
            plt.plot(x, fit1x, label='best fit')
            plt.xlabel('velocity [km/s]')
            plt.ylabel('flux [Jy]')
            plt.title('2Gaussian_model_M1_pixel' + str(ii) + '-' + str(jj) )
            plt.legend()
            plt.show()
        if printfit == True:
            print('##')
            print('2gaussian_fit gx pixel:' + str(ii) + '-' + str(jj))
            report_fit(out)
        
        
        ##USE THE BIC FOR SELECTING THE BEST FIT
        if jj in range1 and ii in range2:
            # if bic_1g < bic_2g and bic_1g < bic_3g  and bic_2g - bic_1g > 2.3 and bic_3g - bic_1g > 2.3:
            bic_min = np.min([bic_1g, bic_1gx])
            if bic_1g == bic_min:    
                flux_map[jj,ii] = np.nansum(fit1z) * dlambda
                flux_map_O[jj,ii] = np.nansum(fit_O) * dlambda
                vel_map[jj,ii] = np.nansum((fit1z*velocity_O)) * dlambda / flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit1z*(velocity_O-vel_map[jj,ii])**2)) * dlambda / flux_map[jj,ii]
                vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                mod[:,jj,ii] = mod1[:,jj,ii]
            #     amperr[jj,ii] = np.sqrt(amp2_1err[jj,ii]**2 + amp2_2err[jj,ii]**2)
            #     velerr[jj,ii] = np.sqrt(cen2_1err[jj,ii]**2 + cen2_2err[jj,ii]**2)
            #     disperr[jj,ii] = np.sqrt(wid2_1err[jj,ii]**2 + wid2_2err[jj,ii]**2)
            elif bic_1gx == bic_min:
                flux_map[jj,ii] = np.nansum(fit1xz) * dlambda 
                flux_map_O[jj,ii] = np.nansum(fit_Ox) * dlambda
                vel_map[jj,ii] = np.nansum((fit1xz*velocity_O)) * dlambda / flux_map[jj,ii]
                vdisp_map[jj,ii] = np.nansum((fit1xz*(velocity_O-vel_map[jj,ii])**2)) * dlambda / flux_map[jj,ii]
                vdisp_map[jj,ii] = vdisp_map[jj,ii]**0.5
                mod[:,jj,ii] = mod1x[:,jj,ii]
            #     amperr[jj,ii] = np.sqrt(amp2x_1err[jj,ii]**2 + amp2x_2err[jj,ii]**2)
            #     velerr[jj,ii] = np.sqrt(cen2x_1err[jj,ii]**2 + cen2x_2err[jj,ii]**2)
            #     disperr[jj,ii] = np.sqrt(wid2x_1err[jj,ii]**2 + wid2x_2err[jj,ii]**2)
        
# flux_map[flux_map_tmp<3.5*error_O] = np.nan
# vel_map[flux_map_tmp < 3.5*error_O] = np.nan
# vdisp_map[flux_map_tmp < 3.5*error_O] = np.nan                 


# Plot            
plt.figure(figsize = (20,6))

plt.subplot(131)
plt.imshow(flux_map_O, origin = 'lower', cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'Flux [Jy]')
plt.title('Flux')
plt.subplot(132)
plt.imshow(vel_map, origin = 'lower', vmin = -400, vmax = 400, cmap ='jet')
plt.colorbar(shrink = 0.7, label='Velocity [km/s]')
plt.title('Velocity')
plt.subplot(133)
plt.imshow(vdisp_map, origin = 'lower',  vmin = 0, vmax=500, cmap = 'jet')
plt.colorbar(shrink = 0.7, label = 'Velocity Dispersion [km/s]')
plt.title('Velocity Dispersion')
plt.suptitle('Moments Maps')

# #----------------------------------------------------------------------------
# Saving the data
nn= 3
hdu = fits.PrimaryHDU(mod)
hdul = fits.HDUList([hdu])
hdul.writeto('model_' + str(nn) + '_1gx' + 'NGC6810_MUSE_OIII' +'.fits', overwrite = True)


hdu = fits.PrimaryHDU(flux_map)
hdul = fits.HDUList([hdu])
hdul.writeto('flux_map_spiral_' + str(nn) +'_1gx' + 'NGC6810_MUSE_OIII' + '.fits', overwrite = True)
hdu = fits.PrimaryHDU(flux_map_O)
hdul = fits.HDUList([hdu])
hdul.writeto('flux_map_O_5008_spiral_' + str(nn) +'_1gx' + 'NGC6810_MUSE_OIII' + '.fits', overwrite = True)
hdu = fits.PrimaryHDU(vel_map)
hdul = fits.HDUList([hdu])
hdul.writeto('vel_map_spiral_'+ str(nn) +'_1gx' + 'NGC6810_MUSE_OIII' + '.fits', overwrite = True)
hdu = fits.PrimaryHDU(vdisp_map)
hdul = fits.HDUList([hdu])
hdul.writeto('vdisp_map_spiral_'+ str(nn) +'_1gx'  + 'NGC6810_MUSE_OIII' + '.fits', overwrite = True)


# hdu = fits.PrimaryHDU(amperr)
# hdul = fits.HDUList([hdu])
# hdul.writeto('flux_err_map_' + str(nn) +'_2gx' + 'NGC5643' + '.fits', overwrite = True)
# hdu = fits.PrimaryHDU(velerr)
# hdul = fits.HDUList([hdu])
# hdul.writeto('vel_err_map_'+ str(nn) +'_2gx' + 'NGC5643' + '.fits', overwrite = True)
# hdu = fits.PrimaryHDU(disperr)
# hdul = fits.HDUList([hdu])
# hdul.writeto('vdisp_err_map_'+ str(nn) +'_2g'  + 'NGC5643' + '.fits', overwrite = True)

# #----------------------------------------------------------------------------




