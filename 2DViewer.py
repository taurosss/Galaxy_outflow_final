#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:35:20 2018

@author: stefano
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import sys
from astropy import wcs

import cube


rcParams.update({'figure.autolayout': False})



class viewer_2d(object):
    def __init__(self, c1, c2, xrange=None, yrange=None, spwlrange=None, imwlrange=None,
                 image=None, image_range=None, component=[], col_comp = None,
                 fluxRange = None, residualRange = None, resPlot = False):
        """
        Shows a given array in a 2d-viewer.
        Input: z, an 2d array.
        x,y coordinters are optional.
        """        
        Nx = c1.cube.shape[2]
        Ny = c1.cube.shape[1]

        


        if xrange == None:
            xrange=[0,Nx]
        if yrange == None:
            yrange=[0,Ny]

        if spwlrange == None:
            self.cube1 = c1.cube
            self.cube2 = c2.cube
            self.wl1 = c1.z
            self.wl2 = c2.z
        else:
            c1.zsel(spwlrange)
            c2.zsel(spwlrange)

            self.cube1 = c1.cube[c1.zmask,:,:]
            self.cube2 = c2.cube[c2.zmask,:,:]
            self.wl1 = c1.z[c1.zmask]
            self.wl2 = c2.z[c2.zmask]
        
        if imwlrange is None:
            im = self.cube1.sum(axis = 0)
        else:
            cube = self.cube1[(self.wl1 >= imwlrange[0]) & (self.wl1 <= imwlrange[1]),:,:]
            im = cube.sum(axis = 0)
        
        if (image is not None):
            im = image
            
        if (image_range is not None) :
            vmin = image_range[0]
            vmax = image_range[1]
        else:
            vmin = None
            vmax = None
            

        if len(component)   == 0:
            self.ncomp = 0
        if len(component)   > 0:
#            print "funziona"
            self.ncomp = len(component)
            self.col_comp = col_comp
            component[0].wavesel(spwlrange)
            tmpwlmask = component[0].wlmask

            Nztmp, Nytmp, Nxtmp = component[0].cube[tmpwlmask,:,:].shape
            self.cube_ncomp = np.zeros([len(component),Nztmp,Nytmp,Nxtmp])
      
            for n in range(len(component)):
                
                self.cube_ncomp[n,:,:,:] = component[n].cube[tmpwlmask,:,:]
#            c2.wavesel(spwlrange)
#            self.cube1 = c1.cube[c1.wlmask,:,:]
#            self.cube2 = c2.cube[c2.wlmask,:,:]
#            self.wl1 = c1.wl[c1.wlmask]
#            self.wl2 = c2.wl[c1.wlmask]
#            
        self.xold = 0
        self.yold = 0
        self.fig=plt.figure(figsize=(15, 5))
        #Doing some layout with subplots:
        #self.fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9, left=0.0)
        self.overview = plt.subplot2grid((3,10),(0,0),rowspan=3,colspan=3)
        self.spec_plot = plt.subplot2grid((3,10),(0,3),rowspan=2,colspan=7)
        self.spec_plot2 = plt.subplot2grid((3,10),(2,3),rowspan=1,colspan=7)
        self.fig.tight_layout()
        
        self.overview.set_xlim(xrange)
        self.overview.set_ylim(yrange)
        self.spec_plot.set_xlim(spwlrange)
        self.fluxRange = fluxRange
        self.residualRange = residualRange
        self.spec_plot2.set_xlim(spwlrange)


        self.overview.imshow(im,interpolation = 'nearest', origin = 'lower',
                             vmin = vmin, vmax = vmax)
        self.dot = self.overview.scatter(0, 0)   
        
        self.resPlot = resPlot
#        #Adding widgets, to not be gc'ed, they are put in a list:
#        
#        cursor=Cursor(self.overview, useblit=True, color='black', linewidth=2 )
#        but_ax=plt.subplot2grid((8,4),(7,0),colspan=1)
#        reset_button=Button(but_ax,'Reset')
#        but_ax2=plt.subplot2grid((8,4),(7,1),colspan=1)
#        legend_button=Button(but_ax2,'Legend')
#        self._widgets=[cursor,reset_button,legend_button]
#        #connect events
##        reset_button.on_clicked(self.clear_xy_subplots)
##        legend_button.on_clicked(self.show_legend)

        self.doPlot(np.round(Nx/2), np.round(Ny/2))

        sys.stdout.flush()
        self.lock = False
        
        # draw if mouse is moved in image
        self.fig.canvas.mpl_connect('motion_notify_event',self.mousemov)

        # draw next pixel if key is pressed
        self.fig.canvas.mpl_connect('key_press_event',self.keypress)
        
        # draw if mouse is pressed
        self.fig.canvas.mpl_connect('button_press_event', self.mouseclick)
##        self.fig.canvas.draw()
        


    def mousemov(self,event):
        """
        What to do, if a click on the figure happens:
            1. Check which axis
            2. Get data coord's.
            3. Plot resulting data.
            4. Update Figure
        """
        if event.inaxes==self.overview and self.lock == False:
            if (np.abs(event.xdata-self.xold)>0.5) |   (np.abs(event.ydata-self.yold)>0.5):
            #Get nearest data
#                plt.gca()
                xpos=np.abs(event.xdata)
                ypos=np.abs(event.ydata)
                self.doPlot(xpos, ypos)

    def mouseclick(self,event):
#        print event
        sys.stdout.flush()
        self.click(event)

    def keypress(self,event):
        if event.inaxes==self.overview:
            xpos = self.xold
            ypos = self.yold       

            if event.key == 'right':
                xpos = xpos+1
            if event.key == 'left':
                xpos = xpos-1
            if event.key == 'up':
                ypos = ypos+1
            if event.key == 'down':
                ypos = ypos-1
            if event.key == 'm':
#                print 'Cursor position locked'
                sys.stdout.flush()
                self.lock = True
            if event.key == 'n':
#                print 'Cursor position unlocked'
                sys.stdout.flush()
                self.lock = False

#            if event.key == 'a':
#                self.overview.imshow(im[1,:,:],interpolation = 'nearest', origin = 'lower',
#                             vmin = vmin, vmax = vmax)


            self.doPlot(xpos, ypos)

    def doPlot(self, xpos, ypos, lab=None):

        # for j in self.spec_plot:
        #     j.lines=[]
        #     plt.draw()
        # for j in self.spec_plot2:
        #     j.lines=[]
        #     plt.draw()
        for j,line in enumerate(self.spec_plot.lines):
            self.spec_plot.lines.pop(j)
            line.remove()
            
        #     j.lines=[]
        #     plt.draw()
        # for j in [self.spec_plot2]:
        #     j.lines=[]
        #     plt.draw()
        
        self.dot.remove()        
        self.dot = self.overview.scatter(xpos, ypos, color='white')   
        
        xpos = xpos.astype(np.int64)
        ypos = ypos.astype(np.int64)

        wl = self.wl1
        spec = self.cube1[:,ypos,xpos]
#        print(self.cube1.shape)
        mod = self.cube2[:,ypos,xpos]
        if self.resPlot == True:
            res = spec-mod
        
        ######### modified by Stefano       17/03/2016   ######
        if self.ncomp != 0:
            spec_ncomp = np.zeros([self.ncomp,len(self.cube_ncomp[0,:,ypos,xpos])])
            for n in range(self.ncomp):
                spec_ncomp[n,:] = self.cube_ncomp[n,:,ypos,xpos]
        ########################################################
        if  self.fluxRange is None: 
            lim = [min(spec)-0.1*max(spec),max(spec)*1.1]
        if  self.fluxRange is not None: 
            lim = [ self.fluxRange[0], self.fluxRange[1]]            
            
        if  self.residualRange is None: 
            limResidual = None
        if  self.residualRange is not None: 
            limResidual = [ self.residualRange[0], self.residualRange[1]]          

        xl1 = self.spec_plot.get_xlim()
        xl2 = self.spec_plot2.get_xlim()
        if xl2 != xl1:
            self.spec_plot2.set_xlim(xl1)            
        
        self.spec_plot.set_title('Pos %s %s' % (xpos, ypos))
        self.spec_plot.set_ylim(lim)
        self.spec_plot2.set_ylim(limResidual)
#                self.x_subplot.clf() 
#                self.spec_plot.cla() 
#                self.overview.figure.cla() 
        c, = self.spec_plot.step(wl,spec,color='black')
        c2, = self.spec_plot.plot(self.wl2,mod,color='red')
        
        if self.resPlot == True:
            c3, = self.spec_plot2.step(wl,res,color='blue')

        ######### modified by Stefano       17/03/2016   ######
        if self.ncomp != 0:
            for n in range(self.ncomp):
                c3, = self.spec_plot.step(wl,spec_ncomp[n,:],color=self.col_comp[n])
        ########################################################

        self.xold = np.round(xpos)
        self.yold = np.round(ypos)

        c.figure.canvas.draw()
        
if __name__=='__main__':

    fitsfile1 = 'file/NGC6810_crop.fits'
    fitsfile2 = 'modelsub7.fits'
    # fitsfile1 = 'Data/Bischetti/PDS456_12CO_cubeVel.fits'

    # fitsfile2 = 'Data/Bischetti/PDS456_12CO_cubeVel_simcube.fits'

    spwlrange= [2.285e11,2.295e11]
#    spwlrange=[6600,7000]

    imwlrange= None
    xrange=None #[100,200]
    yrange=None #[100,200]
#    xrange=None
#    yrange=None
    snlev = 3

    cube1 = cube.Readcube(fitsfile1, beam = False)    
#    cube1.cube /= 0.004273#np.nanmax(cube1.cube)
    cube2 = cube.Readcube(fitsfile2, beam = False) 
    cube2.cube /= np.nanmax(cube2.cube)
    cube1.cube /= np.nanmax(cube1.cube)

    v2d=viewer_2d(cube1, cube2, xrange=xrange, yrange=yrange, imwlrange=imwlrange, 
                  spwlrange=spwlrange,image = None, image_range=None,
                  fluxRange = None, residualRange=None, resPlot = False)
#


