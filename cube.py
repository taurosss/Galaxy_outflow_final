import astropy.io.fits as pyfits
import numpy as np
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.patches import Ellipse
import scipy.ndimage
# from Imaging import estimate_rms
from astropy.modeling import models
class Readcube:
     def __init__(self, filefits, extension = 0, channel = None,
                  beam = True, verbose = False):
            fits = pyfits.open(filefits)
            
            self.filefits = filefits

            naxis = fits[extension].header['NAXIS']
            remove = False
            if naxis > 3:
                if verbose == True:
                  print('WARNING: it is not a datacube')
                  print('naxis =', naxis)
                  print('we remove the axis-0')
                remove = True

            naxis1 = fits[extension].header['NAXIS1']
            naxis2 = fits[extension].header['NAXIS2']
            naxis3 = fits[extension].header['NAXIS3']

            crpix1 = fits[extension].header['CRPIX1']
            crpix2 = fits[extension].header['CRPIX2']
            crpix3 = fits[extension].header['CRPIX3']

            crval1 = fits[extension].header['CRVAL1']
            crval2 = fits[extension].header['CRVAL2']
            crval3 = fits[extension].header['CRVAL3']


            if 'CD1_1' in fits[extension].header:
                cdelt1 = fits[extension].header['CD1_1']
            else:
                cdelt1 = fits[extension].header['CDELT1']

            if 'CD2_2' in fits[extension].header:
                cdelt2 = fits[extension].header['CD2_2']
            else:
                cdelt2 = fits[extension].header['CDELT2']

            if 'CD3_3' in fits[extension].header:
                cdelt3 = fits[extension].header['CD3_3']
            else:
                cdelt3 = fits[extension].header['CDELT3']


            i = 1+np.arange(naxis1)
            j = 1+np.arange(naxis2)
            k = 1+np.arange(naxis3)
            
            x = crval1+cdelt1*(i-crpix1)
            y = crval2+cdelt2*(j-crpix2)
            z = crval3+cdelt3*(k-crpix3)

            self.header = pyfits.getheader(filefits)

            if remove == False: self.cube = fits[extension].data
            if remove == True: self.cube = fits[extension].data[0,:,:,:]

            self.x = x
            self.y = y
            self.z = z

            headerwcs = fits[extension].header #pyfits.getheader(filefits)
            headerwcs['NAXIS'] = 2
            del headerwcs['*3*']
            del headerwcs['*4*']

            self.wcs = wcs.WCS(headerwcs)

            if beam == True:
    
                if 'BMIN' in self.header:
                    width = self.header['BMIN']*3600.
                    height=self.header['BMAJ']*3600.
                    angle=self.header['BPA']
                else:
                    hduTemp = pyfits.open(self.filefits)[1]
                    if channel is None:
                        width =  np.median(hduTemp.data['BMIN'][hduTemp.data['BPA']!=0])
                        height =  np.median(hduTemp.data['BMAJ'][hduTemp.data['BPA']!=0])
                        angle =  np.median(hduTemp.data['BPA'][hduTemp.data['BPA']!=0])
                    else:
                        width =  (hduTemp.data['BMIN'][channel])
                        height =  (hduTemp.data['BMAJ'][channel])
                        angle =  (hduTemp.data['BPA'][channel])                    
                
                self.ALMAbeam = [height,width,angle]
                self.ALMAarea = abs(np.pi*height*width/cdelt1/cdelt2/4.0/np.log(2.0)/3600./3600)
    
                
            
     def zsel(self, zrange, exclude=False, reset=False):
            if not hasattr(self, 'wlmask') or reset: 
                self.zmask = self.z < 0
            if not exclude:
                self.zmask = self.zmask | ( (self.z >=zrange[0]) & (self.z <=zrange[1]) )
            else:
                self.zmask = self.zmask & ( (self.z <=zrange[0]) | (self.z >=zrange[1]) )

     def beam(self, x0, y0, dl, facecolor = 'white', alpha = 0.8, 
              arcsec = False, cell_size = None):
         
            cellSize =  abs(self.header['CDELT2'])
            if arcsec == True: cellSize = 1.0
            if cell_size is not None:  cellSize = cell_size
            if 'BMIN' in self.header:
                width = self.header['BMIN']/cellSize
                height=self.header['BMAJ']/cellSize
                angle=self.header['BPA']
            else:
                hduTemp = pyfits.open(self.filefits)[1]
                width =  np.median(hduTemp.data['BMIN']/cellSize)
                height =  np.median(hduTemp.data['BMAJ']/cellSize)
                angle =  np.median(hduTemp.data['BPA'])
            r = height/2.
            ell = Ellipse(xy=[x0-dl+r+1,y0-dl+r+1], width=width, 
              height=height, angle=angle,
            facecolor = facecolor, alpha =alpha)
        
            # print height,width
            return ell
        
     def draw_beam(self,ax,alma_beam = None, loc='lower left',
                 pad=0.5, borderpad=0.5, prop=None, frameon=False,
                 alpha = 0.5, facecolor = 'white', cell_size = None):
    
        from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox)

        if alma_beam is None:
            width, height, angle = self.ALMAbeam
        else:
            width, height, angle = alma_beam
             
        if cell_size is None:
            cell_size = abs(np.diff(self.x)[0]*3600)
             
        width /= cell_size
        height /= cell_size
        angle += 90.
    
#    print(width, height, angle)
#    
        class AnchoredEllipse(AnchoredOffsetbox):
    
            def __init__(self, transform, width, height, angle, loc,
                     facecolor,frameon, alpha, pad, borderpad,
                     prop=None):

                self._box = AuxTransformBox(transform)
                self.ellipse = Ellipse((0, 0), width, height, angle, 
                                   alpha = alpha, facecolor = facecolor)
                self._box.add_artist(self.ellipse)
                super().__init__(loc, pad=pad, borderpad=borderpad,
                             child=self._box, prop=prop, frameon=frameon)
                
        ae = AnchoredEllipse(ax.transData, width=width, height=height, angle=angle,
                         loc=loc, pad=pad, borderpad=borderpad,
                         frameon=frameon, alpha = alpha, facecolor = facecolor)

        ax.add_artist(ae)


     def cropTarget(self, x0,y0,dl):
            
            x0,y0,dl = int(x0),int(y0),int(dl)
            raTmp, decTmp = self.wcs.wcs_pix2world(x0,y0,0)
            
            cubeTmp = self.cube[:,y0-dl:y0+dl,x0-dl:x0+dl]
            self.cube = cubeTmp

            Ny,Nx = self.cube[0,:,:].shape
            self.wcs.wcs.crpix = [Ny/2.,Nx/2.]
            self.wcs.wcs.crval =[raTmp,decTmp]
         
             
    

def draw_beam(ax,alma_beam = None, loc='lower left',
                 pad=0.5, borderpad=0.5, prop=None, frameon=False,
                 alpha = 0.5, facecolor = 'white', cell_size = None):
    
        from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox)

        if alma_beam is None:
            width, height, angle = [1.,1.,0.]
        else:
            width, height, angle = alma_beam
             
        if cell_size is None:
            cell_size = 1.
             
        width /= cell_size
        height /= cell_size
        angle += 90.
#        print("cell_size = {}".format(cell_size))
#        print(width, height, angle)
#    
        class AnchoredEllipse(AnchoredOffsetbox):
    
            def __init__(self, transform, width, height, angle, loc,
                     facecolor,frameon, alpha, pad, borderpad,
                     prop=None):

                self._box = AuxTransformBox(transform)
                self.ellipse = Ellipse((0, 0), width, height, angle, 
                                   alpha = alpha, facecolor = facecolor)
                self._box.add_artist(self.ellipse)
                super().__init__(loc, pad=pad, borderpad=borderpad,
                             child=self._box, prop=prop, frameon=frameon)
                
        ae = AnchoredEllipse(ax.transData, width=width, height=height, angle=angle,
                         loc=loc, pad=pad, borderpad=borderpad,
                         frameon=frameon, alpha = alpha, facecolor = facecolor)

        ax.add_artist(ae)



class Readimage:
     def __init__(self, filefits, error = False, header = True, 
                  extension = 0, beam = True):
            fits = pyfits.open(filefits)

            naxis = fits[extension].header['NAXIS']

            naxis1 = fits[extension].header['NAXIS1']
            naxis2 = fits[extension].header['NAXIS2']

            if header == True:
                crpix1 = fits[extension].header['CRPIX1']
                crpix2 = fits[extension].header['CRPIX2']
    
                crval1 = fits[extension].header['CRVAL1']
                crval2 = fits[extension].header['CRVAL2']
    
                check = 'CDELT1' in fits[extension].header 
                if check == False:
                    cdelt1 = fits[extension].header['CD1_1']
                    if isinstance(cdelt1, str): cdelt1 = float(cdelt1[1:])
                    cdelt2 = fits[extension].header['CD2_2']
                else :
                    cdelt1 = fits[extension].header['CDELT1']
                    cdelt2 = fits[extension].header['CDELT2']
                

            if header == False:
                crval1 = 0
                crval2 = 0
                cdelt1 = 1
                cdelt2 = 1
                crpix1 = 0                                   
                crpix2 = 0
            
            i = 1+np.arange(naxis1)
            j = 1+np.arange(naxis2)
            
            x = crval1+cdelt1*(i-crpix1)
            y = crval2+cdelt2*(j-crpix2)

#            print len(fits[0].data.shape)
#            self.cube = np.transpose(fits[0].data,(1,2,0))
            if len(fits[extension].data.shape) == 4:
                fits[extension].header['NAXIS'] = 2
                del fits[extension].header['*3*']
                del fits[extension].header['*4*']
                try:
                    del fits[extension].header['']
                except:
                    nothing = True
                del fits[extension].header['HISTORY*']
                self.image = fits[extension].data[0,0,:,:]
            if len(fits[extension].data.shape) == 2:
                self.image = fits[extension].data[:,:]



            self.header = fits[extension].header
            self.x = x
            self.y = y
            if error == True:
                self.error = fits[1].data
#            self.ra = ra
#            self.dec = dec



            headerwcs = pyfits.getheader(filefits,extension)
            headerwcs['NAXIS'] = 2
            del headerwcs['*3*']
            del headerwcs['*4*']

            self.wcs = wcs.WCS(headerwcs)
            if beam == True:
    
                if 'BMIN' in self.header:
                    width = self.header['BMIN']*3600.
                    height=self.header['BMAJ']*3600.
                    angle=self.header['BPA']
                else:
                    hduTemp = pyfits.open(self.filefits)[1]
                    width =  np.median(hduTemp.data['BMIN'][hduTemp.data['BPA']!=0])
                    height =  np.median(hduTemp.data['BMAJ'][hduTemp.data['BPA']!=0])
                    angle =  np.median(hduTemp.data['BPA'][hduTemp.data['BPA']!=0])

                self.ALMAbeam = [height,width,angle]
                self.ALMAarea = abs(np.pi*height*width/cdelt1/cdelt2/4.0/np.log(2.0)/3600./3600)
    
                           
     def beam(self, x0, y0, dl, facecolor = 'white', alpha = 0.8, 
              arcsec = False, cell_size = None):
         
            cellSize =  abs(self.header['CDELT2'])
            if arcsec == True: cellSize = 1.0
            if cell_size is not None:  cellSize = cell_size

            if 'BMIN' in self.header:
                width = self.header['BMIN']/cellSize
                height=self.header['BMAJ']/cellSize
                angle=self.header['BPA']
            else:
                hduTemp = pyfits.open(self.filefits)[1]
                width =  np.median(hduTemp.data['BMIN']/cellSize)
                height =  np.median(hduTemp.data['BMAJ']/cellSize)
                angle =  np.median(hduTemp.data['BPA'])
            r = height/2.
            ell = Ellipse(xy=[x0-dl+r+1,y0-dl+r+1], width=width, 
              height=height, angle=angle,
            facecolor = facecolor, alpha =alpha)

            # print height,width
            return ell
       

     def cropTarget(self, x0,y0,dl):
            
            x0,y0,dl = int(x0),int(y0),int(dl)
            raTmp, decTmp = self.wcs.wcs_pix2world(x0,y0,0)
            
            imageTmp = self.image[y0-dl:y0+dl,x0-dl:x0+dl]
            self.image = imageTmp
#            
#            crpixTmp = np.copy(self.wcs.wcs.crpix)
#            print crpixTmp[0]-y0,crpixTmp[1]-x0,crpixTmp
            Ny,Nx = self.image.shape
            self.wcs.wcs.crpix = [Nx/2.,Ny/2.]
            self.wcs.wcs.crval =[raTmp,decTmp]
         
    
     def find_centroid(self):
         
         j,i = np.where(self.image == np.nanmax(self.image))
         self.y0 = self.y[j]*1.0
         self.x0 = self.x[i]*1.0
         
     # def f_xydist(self):
        
     #    xx,yy = np.meshgrid(self.x-self.x0, self.y-self.y0)
     #    self.xydist = np.hypot(np.hstack(xx), np.hstack(yy))
     #    self.flux = np.hstack(self.image)
     #    self.weights = 1./(np.zeros_like(self.flux)+estimate_rms.estimate_rms(self.image,niter = 20))**2
     #    self.xydist *= 3600. #arcsec
    
     def f_xybin(self, xybin_size, psf = False, **kwargs):
     
        """
        Compute the intervals (bins) of the xy-distances given the size of the bin d.
        Parameters
        ----------
        xybin_size : float
            Bin size in units of the wavelength.
        Note
        ----
        To compute the weights, we do not need to divide by the weight_corr factor since it cancels out when we compute
        """
        
        if psf:
            g2d = models.Gaussian2D(amplitude=1, x_mean=0., y_mean=0., 
            x_stddev=self.ALMAbeam[0]/2.355/3600., y_stddev=self.ALMAbeam[1]/2.355/3600., 
            theta=np.deg2rad(self.ALMAbeam[1]))
            xx,yy = np.meshgrid(self.x-self.x0, self.y-self.y0)
            g2d_image = g2d(xx,yy)
            
            self.psf = np.hstack(g2d_image)
        
        
        
        self.f_xydist()
        
        self.nbins = np.ceil(self.xydist.max()/xybin_size).astype('int')
        self.bin_xydist = np.zeros(self.nbins)
        self.bin_weights = np.zeros(self.nbins)
        self.bin_count = np.zeros(self.nbins, dtype='int')
        self.bin_count_cum = np.zeros(self.nbins, dtype='int')
        self.bin_flux_cum =  np.zeros(self.nbins)
        self.bin_flux_cum_err =  np.zeros(self.nbins)

        self.xy_intervals = []
        self.xy_intervals_cum = []

        self.xy_bin_edges = np.arange(self.nbins+1, dtype='float64')*xybin_size

        for i in range(self.nbins):
            xy_interval = np.where((self.xydist >= self.xy_bin_edges[i]) &
                                   (self.xydist < self.xy_bin_edges[i+1]))
            self.bin_count[i] = len(xy_interval[0])

            if self.bin_count[i] != 0:
                self.bin_xydist[i] = self.xydist[xy_interval].sum()/self.bin_count[i]
                self.bin_weights[i] = np.sum(self.weights[xy_interval])
                
            else:
                self.bin_xydist[i] = self.xy_bin_edges[i]+0.5*xybin_size
                
            self.xy_intervals.append(xy_interval)
            
            xy_interval_cum = np.where((self.xydist < self.xy_bin_edges[i]))
            self.bin_count_cum[i] = len(xy_interval_cum[0])
            self.bin_flux_cum[i] = np.nansum(self.flux[xy_interval_cum])/self.ALMAarea
            self.bin_flux_cum_err[i] = self.weights[0]**-0.5*(self.bin_count_cum[i]/self.ALMAarea)**0.5
            

        self.bin_flux, self.bin_flux_err = self.f_bin_quantity(self.flux, **kwargs)
        if psf:
            self.bin_psf, self.bin_psf_err = self.f_bin_quantity(self.psf, **kwargs)


     def f_bin_quantity(self, x, use_std=False, norm = True):
        """
        Compute bins of the quantity x based on the intervals of the uv-distances of the current Uvtable.
        To compute the uv-distances use Uvtable.compute_uvdist() and to compute the intervals use Uvtable.compute_uv_intervals().
        Parameters
        ----------
        x : array-like
            Quantity to be binned.
        use_std : bool, optional
            If provided, the error on each bin is computed as the standard deviation divided by sqrt(npoints_per_bin).
        Returns
        -------
        bin_x, bin_x_err : array-like
            Respectively the bins and the bins error of the quantity x.
        """
        bin_x, bin_x_err = np.zeros(self.nbins), np.zeros(self.nbins)

        for i in range(self.nbins):

            if self.bin_count[i] != 0:
                bin_x[i] = np.sum(x[self.xy_intervals[i]]*self.weights[self.xy_intervals[i]])/self.bin_weights[i]

                if use_std is True:

                    bin_x_err[i] = np.std(x[self.xy_intervals[i]])
                else:
                    bin_x_err[i] = self.weights[0]**-0.5#1./np.sqrt(self.bin_weights[i])

        if norm:
            bin_x_err /= np.nanmax(bin_x)
            bin_x /= np.nanmax(bin_x)
            
        return bin_x, bin_x_err        
             
    



class Readspec:
     def __init__(self, filefits, extension = 0):
            fits = pyfits.open(filefits)

            naxis = fits[extension].header['NAXIS']

            naxis1 = fits[extension].header['NAXIS1']

            crpix1 = fits[extension].header['CRPIX1']

            crval1 = fits[extension].header['CRVAL1']

            check = 'CDELT1' in fits[extension].header 
            if check == False:
                cdelt1 = fits[extension].header['CD1_1']
                if isinstance(cdelt1, str): cdelt1 = float(cdelt1[1:])
            else :
                cdelt1 = fits[extension].header['CDELT1']
                
#            cunit1 = fits[0].header['CUNIT1']
#            ctype1 = fits[0].header['CTYPE1']
            #ra = fits[0].header['RA']
            #dec = fits[0].header['DEC']
            
            i = 1+np.arange(naxis1)
            x = crval1+cdelt1*(i-crpix1)

            self.header = fits[extension].header
            self.spec = fits[extension].data

            self.z = x
#            self.ra = ra
#            self.dec = dec




class Region:
    def __init__(self, ds9regionfile, imagewcs):
        
        ds9region = np.loadtxt(ds9regionfile,dtype='str', skiprows = 3)

#        print ds9region
        Nreg = len(ds9region)

        regline = []
        shape = []
        for i in range(Nreg):

            reg = ds9region[i]
            k0  = reg.find('(')+1
            shapetmp = reg[:k0-1]
            shape.append(shapetmp)

            if shapetmp == 'polygon':
                j = 0
                ra = []
                dec = []
                while(j!=1):             
                    k1 = reg[k0:].find(',')
                    ratmp = reg[k0:k1+k0]
                    k2 = reg[k0+k1+1:].find(',')
                    if k2 == -1 :
                        k2 = reg[k0+k1+1:].find(')')
                        j = 1
                    dectmp = reg[k0+k1+1:k0+k2+k1+1]
                    k0 = k0+k2+k1+2
                    ra.append(ratmp)
                    dec.append(dectmp)
                    
                coord = SkyCoord(ra, dec, frame='icrs',unit=(u.hourangle, u.deg))
                xp,yp = imagewcs.all_world2pix(coord.ra.deg,coord.dec.deg,0)
    
                regline.append([xp,yp])
            
            if shapetmp == 'circle':
                k1 = reg[k0:].find(',')
                ratmp = reg[k0:k1+k0]
                k2 = reg[k0+k1+1:].find(',')
                dectmp = reg[k0+k1+1:k0+k2+k1+1]
                k3 = reg[k0+k2+k1+1:].find(')')
                radius = float(reg[k0+k2+k1+2:k0+k2+k1+k3])
                
                
                coord = SkyCoord(ratmp, dectmp, frame='icrs',unit=(u.hourangle, u.deg))
                xp,yp = imagewcs.all_world2pix(coord.ra.deg,coord.dec.deg,0)
                regline.append([xp,yp,radius])

    
        self.region = regline
        self.nreg = Nreg
        self.shape = shape
        

#def rebin(filefits_input, new_axis = None, rebinning = None, axis = None):
    
#    c = cube.Readcube()


# if __name__ == "__main__" :
    
#     import matplotlib.pyplot as plt
#     from T03 import drizzle 
# #    print "TEST"
    
#     path = 'Data/BDF3299/fits/'
#     fitsfile = path+'BDF-3299_2016_cii.fits'
    
#     rebinning = None,
#     axis = None
#     output = None
    
#     clean_header = True
    
#     c = Readcube(fitsfile, extension = 0, beam = False)
#     Nz,Ny,Nx = c.cube.shape
#     header = c.header.copy()
    
#     if clean_header:
#         del header['*4*']
    
    
#     datacube = scipy.ndimage.zoom(c.cube, [0.25,1,1])
#     pyfits.writeto(fitsfile[:-4]+'_rebin.fits',datacube )
    
    
    
    
#     wlDrz = np.arange(c.z[0],c.z[-1],4*np.mean(np.diff(c.z)))

#     new_cube = np.zeros([len(wlDrz),Ny,Nx])
#     header['NAXIS3'] = len(wlDrz)


#     for j in range(Ny):
#         for i in range(Nx):
            
#             print(j,i)
#             spec_tmp = c.cube[:,j,i]
#             spec_new, var, weight = drizzle.specdrizzle(c.z[::-1], spec_tmp[::-1], wlDrz[::-1], weight = None, var = None,
#                     infValue = True)
            
            
#             new_cube[:,j,i] = spec_new[::-1]
        
#     plt.step(c.z,spec_tmp)    
#     plt.step(wlDrz,spec_new[::-1])    
        
#     pyfits.writeto(fitsfile[:-4]+'_rebin2.fits',new_cube,header)
        
    
    
# class read_mos_etc_sim:
    
#     def __init__(self, filefits):
        
#         fits = pyfits.open(filefits)
#         self.filefits = filefits
#         self.image = fits[0].data
#         self.y = fits[2].data['y']
#         self.wl = fits[1].data['x']
#         self.center = fits[3].data['center']
#         self.off = fits[3].data['off']
#         self.width = fits[3].data['width']
#         self.flux = fits[3].data['strength']
# #  

# class xshooter:
#      def __init__(self, fitsfile, wl0 = None, dwl0 = None,
#                   edges = 0):
            
#         hdu = pyfits.open(fitsfile)
        
#         naxis1 = hdu[0].header['NAXIS1']
#         naxis2 = hdu[0].header['NAXIS2']
#         crpix1 = hdu[0].header['CRPIX1']
#         crpix2 = hdu[0].header['CRPIX2']
#         crval1 = hdu[0].header['CRVAL1']
#         crval2 = hdu[0].header['CRVAL2']
#         cdelt1 = hdu[0].header['CDELT1']
#         cdelt2 = hdu[0].header['CDELT2']
        
#         i = 1+np.arange(naxis1)
#         j = 1+np.arange(naxis2)
            
#         x = crval1+cdelt1*(i-crpix1)
#         y = crval2+cdelt2*(j-crpix2)

#         self.header = pyfits.getheader(fitsfile)
#         self.fitsfile = fitsfile
  
#         self.wl = x
#         self.off = y
                
#         self.image = hdu[0].data
#         self.err = hdu[1].data
#         self.quality = hdu[2].data
        
#         header =  hdu[0].header.copy()
        
#         if (wl0 is not None) and (dwl0 is not None):

#             w_sel = np.where(abs(x-wl0)<=dwl0)[0]
#             self.wl = x[w_sel]
#             self.image = hdu[0].data[:,w_sel]
#             self.err = hdu[1].data[:,w_sel]
#             self.quality = hdu[2].data[:,w_sel]
            
#             header['NAXIS1'] = len(self.wl)
#             header['CRVAL1'] = self.wl[0]

#         if edges>0:

#             self.off = self.off[edges:-edges]
#             self.image = self.image[edges:-edges,:]
#             self.err =  self.err[edges:-edges,:]
#             self.quality =  self.quality[edges:-edges,:]
            
#             header['NAXIS2'] = len(self.off)
#             header['CRVAL2'] = self.off[0]
               
        
#         self.header = header
       
#         self.wcs = wcs.WCS(self.header)  
#         self.dwl = np.diff(x).mean()
#         self.doff = np.diff(y).mean()
        