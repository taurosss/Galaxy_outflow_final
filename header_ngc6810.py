#import python libraries
import numpy as np 
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.modeling import models, fitting
from scipy.ndimage import gaussian_filter
import os 
import time
from astropy.nddata import Cutout2D
from scipy import ndimage

path = os.path.dirname(os.path.abspath('__file__'))


filefits_hst_blue_filter = path+'/file/NGC6810.fits'
#filefits_hst_green_filter = path+'/file/1250um.fits'
#filefits_hst_red_filter = path+'/file/1600um.fits'


hst_blue = fits.open(filefits_hst_blue_filter)[0]
#hst_green = fits.open(filefits_hst_green_filter)[0]
#hst_red = fits.open(filefits_hst_red_filter)[0]


hst_blue.header


hst_blue_wcs = wcs.WCS(hst_blue.header)
#hst_green_wcs = wcs.WCS(hst_green.header)
#hst_red_wcs = wcs.WCS(hst_red.header)


plt.figure(figsize = (16,8))
ax1 = plt.subplot(131,projection=hst_blue_wcs)
plt.imshow(hst_blue.data,origin = 'lower', vmax = np.percentile(hst_blue.data,99))
dec = ax1.coords[1]
dec.display_minor_ticks(True)
ra = ax1.coords[0]
ra.set_ticklabel()
ra.display_minor_ticks(True)
#ax2 = plt.subplot(132,projection=hst_green_wcs)
# plt.imshow(hst_green.data,origin = 'lower', vmax = np.percentile(hst_green.data,99))
# dec = ax2.coords[1]
# dec.display_minor_ticks(True)
# ra = ax2.coords[0]
# ra.set_ticklabel()
# ra.display_minor_ticks(True)
# ax3 = plt.subplot(133,projection=hst_red_wcs)
# plt.imshow(hst_red.data,origin = 'lower', vmax = np.percentile(hst_red.data,99))
# dec = ax3.coords[1]
# dec.display_minor_ticks(True)
# ra = ax3.coords[0]
# ra.set_ticklabel()
# ra.display_minor_ticks(True)


image = np.copy(hst_blue.data) 
#play with the other data 
#image = np.copy(hst_green.data) 
#image = np.copy(hst_red.data) 


print("min value = {}".format(np.min(image)))
print("max value = {}".format(np.max(image)))

time.sleep(0.1)
min_value = float(input("min value of the histogram? "))
max_value = float(input("max value of the histogram? "))
n_bins = int(input("number of bins? "))

bins = np.linspace(min_value, max_value, n_bins)
hist, bins_edge = np.histogram(image,bins = bins, range = [min_value,max_value])
bins_centers = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

std = np.std(image)
print("standard deviation = {}".format(std))

g_init = models.Gaussian1D(amplitude=np.max(hist), mean=0, stddev=std/10.,
                            bounds={"stddev": (0.0001, 1)})
#g_init.fixed['mean'] = True
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init,bins_centers ,hist)



plt.figure(figsize = (16,8))
plt.subplot(121)
plt.step(bins_centers,hist)
plt.plot(bins_centers,g(bins_centers), color = 'red')
plt.subplot(122)
plt.step(bins_centers,hist)
plt.plot(bins_centers,g(bins_centers), color = 'red')
plt.yscale('log')
plt.ylim(0.1,np.max(hist)*1.1)


sigma = g.stddev.value
print("noise level is {}".format(sigma))
#


hst_blue_noise_level = 0.145
#hst_green_noise_level = 0.00233
#hst_red_noise_level = 0.00180

image_snr_blue = hst_blue.data/hst_blue_noise_level
#image_snr_green = hst_green.data/hst_green_noise_level
#image_snr_red = hst_red.data/hst_red_noise_level


#overplot noise level contours on image
#contours are in steps of 4σ, starting at 1σ.


plt.figure(figsize = (16,8))
plt.subplot(131)
plt.imshow(hst_blue.data,origin = 'lower', vmax = np.percentile(hst_blue.data,99))

#zoom-in
cutout = Cutout2D(hst_blue.data, (332,270), (40,40))
cutout.plot_on_original(color='white')
ax2 = plt.subplot(132)
plt.imshow(cutout.data,origin = 'lower', vmax = np.percentile(hst_blue.data,99))
plt.contour(cutout.data,levels = hst_blue_noise_level*np.arange(1,100,4), colors = 'white')
plt.show()


#define a signal-to-noise threshold for target identification for the blue image
sn_threshold_blue = 5

#find pixels above the threshold
mask_blue = hst_blue.data>(sn_threshold_blue*hst_blue_noise_level)


#plot mask_blue and zoom-in
#0 = pixel below threshold
#1 = pixel above threshold

plt.figure(figsize = (16,8))
plt.subplot(131)
plt.imshow(mask_blue,origin = 'lower')
cutout = Cutout2D(mask_blue, (332,270), (40,40))

cutout.plot_on_original(color='white')
ax2 = plt.subplot(132)
plt.imshow(cutout.data,origin = 'lower',vmin =0, vmax = 1)
plt.colorbar()
plt.show()


#differentiate separate objects
map_labels_blue, n_labels_blue = ndimage.label(mask_blue)

#show example 
plt.figure(figsize = (16,8))
plt.subplot(121)
plt.imshow(map_labels_blue,origin = 'lower', cmap = 'jet')

cutout = Cutout2D(map_labels_blue, (332,270), (40,40))
cutout.plot_on_original(color='white')
ax2 = plt.subplot(122)
plt.imshow(cutout.data,origin = 'lower', vmin = 0, vmax= np.nanmax(map_labels_blue), 
          cmap = 'jet')
plt.colorbar()



#determine the centroid of all identified objects
blue_pixels = ndimage.measurements.center_of_mass(hst_blue.data, map_labels_blue,
                                                   np.arange(1,n_labels_blue+1))
blue_pixels_arr = np.asarray(blue_pixels)

#turn pixel coordinates in to astronomical coordinates (x,y) -> (RA,Dec)
blue_coord = hst_blue_wcs.all_pix2world(blue_pixels_arr[:,1],blue_pixels_arr[:,0],0)

i = 150
print("Example")
print("target {}".format(i))
print("x0 = {}  y0 = {}".format(blue_pixels_arr[i,1],blue_pixels_arr[i,0]))
print("RA = {}deg Dec = {}deg".format(blue_coord[0][i],blue_coord[1][i]))
print("find this galaxy in the observation at 800um with DS9 ")


#generate a catalougue 
#np.savetxt(path+'/file/blue_catalogue.txt', np.transpose(blue_coord))



#same process for observation in the red filter
# sn_threshold = 10

# mask_red = hst_red.data>(sn_threshold*hst_red_noise_level)
# map_labels_red, n_labels_red = ndimage.label(mask_red)

# red_pixels = ndimage.measurements.center_of_mass(hst_red.data, map_labels_red,
#                                                    np.arange(1,n_labels_red+1))
# red_pixels_arr = np.asarray(red_pixels)
# red_coord = hst_red_wcs.all_pix2world(red_pixels_arr[:,1],red_pixels_arr[:,0],0)

# np.savetxt(path+'/file/red_catalogue.txt', np.transpose(red_coord))




# #same process for observation in the green filter
# sn_threshold = 10
# mask_green = hst_green.data>(sn_threshold*hst_green_noise_level)
# map_labels_green, n_labels_green = ndimage.label(mask_green)

# green_pixels = ndimage.measurements.center_of_mass(hst_green.data, map_labels_green,
#                                                    np.arange(1,n_labels_green+1))
# green_pixels_arr = np.asarray(green_pixels)
# green_coord = hst_green_wcs.all_pix2world(green_pixels_arr[:,1],green_pixels_arr[:,0],0)

# np.savetxt(path+'/file/green_catalogue.txt', np.transpose(green_coord))



