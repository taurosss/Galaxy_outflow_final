"""
Created on Mon Feb 15 16:41:48 2021
@author: dkrajnov
Potsdam

exec(open('/Users/dkrajnov/WORK/python/kinemetry/run_kinemetry_examples.py').read())

These examples illustrate how to run kinemetry on velocity and velocity
dispersion maps, as well as galaxy images. Copy the top statement in the ipyhon 
prompt to run the full script. 

"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits 
from plotbin.plot_velfield import plot_velfield

from matplotlib.patches import Ellipse

import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

from kinemetry import kinemetry
import kinemetry as kin

import time
from os import path


#----------------------------------------------------------------------------
def plot_kinemetry_profiles_velocity(k, fitcentre=False, name=None):
    """
    Based on the kinemetry results (passed in k), this routine plots radial
    profiles of the position angle (PA), flattening (Q), k1 and k5 terms.
    Last two plots are for X0,Y0 and systemic velocity
    
    """
    
    k0 = k.cf[:,0]
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)
    k5 = np.sqrt(k.cf[:,5]**2 + k.cf[:,6]**2)
    k51 = k5/k1
    erk1 = (np.sqrt( (k.cf[:,1]*k.er_cf[:,1])**2 + (k.cf[:,2]*k.er_cf[:,2])**2 ))/k1
    erk5 = (np.sqrt( (k.cf[:,5]*k.er_cf[:,5])**2 + (k.cf[:,6]*k.er_cf[:,6])**2 ))/k5
    erk51 = ( np.sqrt( ((k5/k1) * erk1)**2 + erk5**2  ) )/k1 
    

    fig,ax =plt.subplots(figsize=(7,8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1]) 
    

    ax1 = plt.subplot(gs[0])
    ax1.errorbar(k.rad, k.pa, yerr=[k.er_pa, k.er_pa], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax1.set_ylabel('PA [deg]', fontweight='bold')
    if name:
        ax1.set_title(name, fontweight='bold')

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    ax1.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)
    ax1.get_xaxis().set_ticklabels([])

    ax2 = plt.subplot(gs[1])
    ax2.errorbar(k.rad, k.q, yerr=[k.er_q, k.er_q], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax2.set_ylabel('Q ', fontweight='bold')
    #ax2.set_xlabel('R [arsces]')
    ax2.set_ylim(0,1)
    if fitcentre:
        ax2.set_title('Velocity, fit centre', fontweight='bold')
    else:
        ax2.set_title('Velocity, fixed centre', fontweight='bold')


    ax2.tick_params(axis='both', which='both', top=True, right=True)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.yaxis.set_tick_params(length=6)
    ax2.xaxis.set_tick_params(width=2)
    ax2.yaxis.set_tick_params(width=2)
    ax2.xaxis.set_tick_params(length=6)
    ax2.tick_params(which='minor', length=3)
    ax2.tick_params(which='minor', width=1)
    ax2.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax2.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
    ax2.get_xaxis().set_ticklabels([])


    
    ax3 = plt.subplot(gs[2])
    ax3.errorbar(k.rad, k1, yerr=[erk1, erk1], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax3.set_ylabel('$k_1$ [km/s]', fontweight='bold')

    ax3.tick_params(axis='both', which='both', top=True, right=True)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.yaxis.set_tick_params(length=6)
    ax3.xaxis.set_tick_params(width=2)
    ax3.yaxis.set_tick_params(width=2)
    ax3.xaxis.set_tick_params(length=6)
    ax3.tick_params(which='minor', length=3)
    ax3.tick_params(which='minor', width=1)
    ax3.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax3.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(2)
    ax3.get_xaxis().set_ticklabels([])


    ax4 = plt.subplot(gs[3])
    ax4.errorbar(k.rad, k51, yerr=[erk51, erk51], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax4.set_ylabel('$k_{51}$', fontweight='bold')
    ax4.set_xlabel('R [arsces]', fontweight='bold')
    
    ax4.tick_params(axis='both', which='both', top=True, right=True)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.yaxis.set_tick_params(length=6)
    ax4.xaxis.set_tick_params(width=2)
    ax4.yaxis.set_tick_params(width=2)
    ax4.xaxis.set_tick_params(length=6)
    ax4.tick_params(which='minor', length=3)
    ax4.tick_params(which='minor', width=1)
    ax4.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax4.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax4.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax4.spines[axis].set_linewidth(2)
    ax4.get_xaxis().set_ticklabels([])
        
    ax5 = plt.subplot(gs[4])
    ax5.errorbar(k.rad, k.xc, yerr=k.er_xc, fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3, label='Xc')
    ax5.errorbar(k.rad, k.yc, yerr=k.er_yc, fmt='--o', mec='k', mew=1.2, color='salmon', mfc='salmon', capsize=3, label='Yc')
    ax5.set_ylabel('$X_c, Y_c$ [arsces]', fontweight='bold')
    ax5.set_xlabel('R [arsces]', fontweight='bold')
    ax5.legend()

    ax5.tick_params(axis='both', which='both', top=True, right=True)
    ax5.tick_params(axis='both', which='major', labelsize=10)
    ax5.yaxis.set_tick_params(length=6)
    ax5.xaxis.set_tick_params(width=2)
    ax5.yaxis.set_tick_params(width=2)
    ax5.xaxis.set_tick_params(length=6)
    ax5.tick_params(which='minor', length=3)
    ax5.tick_params(which='minor', width=1)
    ax5.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax5.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax5.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax5.spines[axis].set_linewidth(2)
    
    
    ax6 = plt.subplot(gs[5])
    ax6.errorbar(k.rad, k0, yerr=k.er_cf[:,0], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax6.hlines(np.median(k0),5,20,linestyles='dashed', colors='skyblue', label='median $V_{sys}$')
    ax6.set_ylabel('V$_{sys}$ [km/s]', fontweight='bold')
    ax6.set_xlabel('R [arsces]', fontweight='bold')
    ax6.legend()
    
    ax6.tick_params(axis='both', which='both', top=True, right=True)
    ax6.tick_params(axis='both', which='major', labelsize=10)
    ax6.yaxis.set_tick_params(length=6)
    ax6.xaxis.set_tick_params(width=2)
    ax6.yaxis.set_tick_params(width=2)
    ax6.xaxis.set_tick_params(length=6)
    ax6.tick_params(which='minor', length=3)
    ax6.tick_params(which='minor', width=1)
    ax6.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax6.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax6.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax6.spines[axis].set_linewidth(2)
        
    fig.tight_layout()



#----------------------------------------------------------------------------
def plot_kinemetry_maps(xbin, ybin, velbin, k, sigma=False):
    """
    Based on the kinemetry results (k) and original coordinates (xbin,ybin) and 
    the analysed moment (i.e. velocity), this routine plots the original moment 
    (i.e. velocity) map with overplotted best fitted ellispes, reconstructed 
    (rotation) map and a map based on the full Fourier analysis.
    
    """

    
    k0 = k.cf[:,0]
    k1 = np.sqrt(k.cf[:,1]**2 + k.cf[:,2]**2)

    vsys=np.median(k0)
    if sigma:
        mx=np.max(k0)
        mn=np.min(k0)    
        vsys=0
    else:
        mx=np.max(k1)
        mn=-mx
        vsys=np.median(k0)

    
    tmp=np.where(k.velcirc < 123456789)

    fig,ax =plt.subplots(figsize=(12,4))

    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1.06]) 

    ax1 = plt.subplot(gs[0])
    im1 = plot_velfield(xbin, ybin, velbin - vsys, colorbar=False, label='km/s', nodots=True, vmin=mn, vmax=mx)
    ax1.plot(k.Xellip, k.Yellip, ',', label ='ellipse locations')
    ax1.set_ylabel('arcsec', fontweight='bold')
    ax1.set_xlabel('arcsec', fontweight='bold')
    if sigma:
        ax1.set_title('$\sigma$', fontweight='bold')
    else:
        ax1.set_title('V', fontweight='bold')
    ax1.legend()

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)



    ax1 = plt.subplot(gs[1])
    im1 = plot_velfield(xbin[tmp], ybin[tmp], k.velcirc[tmp]- vsys, colorbar=False, label='km/s', nodots=True, vmin=mn, vmax=mx)
    ax1.set_xlabel('arcsec', fontweight='bold')
    if sigma:
        ax1.set_title('$\sigma_0$', fontweight='bold')
    else:
        ax1.set_title('V$_{disk}$', fontweight='bold')
    ax1.plot(k.xc, k.yc, '+', label='(Xc,Yc)')
    ax1.legend()

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)


    ax1 = plt.subplot(gs[2])
    im1 = plot_velfield(xbin[tmp], ybin[tmp], k.velkin[tmp]-vsys, colorbar=False, label='km/s', nodots=True, vmin=mn, vmax=mx)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb=fig.colorbar(im1, cax=cax)
    cb.ax.tick_params(labelsize=12, width=2)
    cb.set_ticks([mn,0,mx],update_ticks=True)
    
    for axis in ['top','bottom','left','right']:
        cb.ax.spines[axis].set_linewidth(5)
    ax1.set_xlabel('arcsec', fontweight='bold')
    if sigma:
        ax1.set_title('$\sigma_{kin}$', fontweight='bold')
        cb.set_label('$\sigma [km/s]', fontweight='bold')
    else:
        ax1.set_title('V$_{kin}$', fontweight='bold')
        cb.set_label('V [km/s]', fontweight='bold')


    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)

    fig.tight_layout()



#----------------------------------------------------------------------------
def plot_kinemetry_profiles_sigma(k, fitcentre=False, name=None, photo=False):

    k0 = k.cf[:,0]
    er_k0 = k.er_cf[:,0]
    b2 = k.cf[:,4]
    er_b2 = k.er_cf[:,4]
    b4 = k.cf[:,8]
    er_b4 = k.er_cf[:,8]

    k20 = b2/k0
    er_k20 = np.sqrt( (er_b2/k0)**2 + (er_k0*(b2/k0**2))**2  )
    k40 = b4/k0
    er_k40 = np.sqrt( (er_b4/k0)**2 + (er_k0*(b4/k0**2))**2  )

    fig,ax =plt.subplots(figsize=(7,9))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1,1,1]) 

    ax1 = plt.subplot(gs[0])

    ax1.errorbar(k.rad, k.pa, yerr=[k.er_pa, k.er_pa], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3, label='fit cen')
    ax1.set_ylabel('PA [deg]', fontweight='bold')
    if name:
        ax1.set_title(name, fontweight='bold')
    ax1.legend()
    if photo:
        ax1.set_xscale('log')

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.yaxis.set_tick_params(length=6)
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    ax1.xaxis.set_tick_params(length=6)
    ax1.tick_params(which='minor', length=3)
    ax1.tick_params(which='minor', width=1)
    ax1.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)
    ax1.get_xaxis().set_ticklabels([])


    ax2 = plt.subplot(gs[1])
    ax2.errorbar(k.rad, k.q, yerr=[k.er_q, k.er_q], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax2.set_ylabel('Q ', fontweight='bold')
    ax2.set_ylim(0,1)
    if fitcentre:
        if photo:
            ax2.set_title('photometry, fit centre', fontweight='bold')
            ax2.set_xscale('log')
        else:
            ax2.set_title(r'$\sigma$, fit centre', fontweight='bold')
    else:
        if photo:
            ax2.set_title('photometry, fixed centre', fontweight='bold')
            ax2.set_xscale('log')
        else:
            ax2.set_title(r'$\sigma$, fixed centre', fontweight='bold')



    ax2.tick_params(axis='both', which='both', top=True, right=True)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.yaxis.set_tick_params(length=6)
    ax2.xaxis.set_tick_params(width=2)
    ax2.yaxis.set_tick_params(width=2)
    ax2.xaxis.set_tick_params(length=6)
    ax2.tick_params(which='minor', length=3)
    ax2.tick_params(which='minor', width=1)
    ax2.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax2.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2)
    ax2.get_xaxis().set_ticklabels([])


    ax3 = plt.subplot(gs[2])
    if photo:
        ax3.errorbar(k.rad, k0, yerr=k.er_cf[:,0], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
        ax3.set_ylabel(r'log$_{10} a_0$', fontweight='bold')   
        ax3.set_xscale('log')
        ax3.set_yscale('log')
    else:
        ax3.errorbar(k.rad, k0, yerr=k.er_cf[:,0], fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
        ax3.set_ylabel(r'$\sigma_0$ [km/s]', fontweight='bold')

    ax3.tick_params(axis='both', which='both', top=True, right=True)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.yaxis.set_tick_params(length=6)
    ax3.xaxis.set_tick_params(width=2)
    ax3.yaxis.set_tick_params(width=2)
    ax3.xaxis.set_tick_params(length=6)
    ax3.tick_params(which='minor', length=3)
    ax3.tick_params(which='minor', width=1)
    ax3.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax3.yaxis.get_major_ticks():    
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(2)
    ax3.get_xaxis().set_ticklabels([])
  
    ax4 = plt.subplot(gs[3])
    ax4.errorbar(k.rad, k20, yerr=er_k20, fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax4.set_ylabel('$b_2/b_0$ [km/s]', fontweight='bold')
    if photo:
        ax4.set_xscale('log')

    ax4.tick_params(axis='both', which='both', top=True, right=True)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.yaxis.set_tick_params(length=6)
    ax4.xaxis.set_tick_params(width=2)
    ax4.yaxis.set_tick_params(width=2)
    ax4.xaxis.set_tick_params(length=6)
    ax4.tick_params(which='minor', length=3)
    ax4.tick_params(which='minor', width=1)
    ax4.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax4.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax4.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax4.spines[axis].set_linewidth(2)
    ax4.get_xaxis().set_ticklabels([])


    ax5 = plt.subplot(gs[4])
    ax5.errorbar(k.rad, k40, yerr=er_k40, fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3)
    ax5.set_ylabel('$b_4/\sigma_0$', fontweight='bold')
    if photo:
        ax5.set_xscale('log')
        ax5.set_xlabel('log$_{10}$ R [arsces]', fontweight='bold')
    else:
        ax5.set_xlabel('R [arsces]', fontweight='bold')


    ax5.tick_params(axis='both', which='both', top=True, right=True)
    ax5.tick_params(axis='both', which='major', labelsize=10)
    ax5.yaxis.set_tick_params(length=6)
    ax5.xaxis.set_tick_params(width=2)
    ax5.yaxis.set_tick_params(width=2)
    ax5.xaxis.set_tick_params(length=6)
    ax5.tick_params(which='minor', length=3)
    ax5.tick_params(which='minor', width=1)
    ax5.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax5.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax5.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax5.spines[axis].set_linewidth(2)


    ax6 = plt.subplot(gs[5])
    ax6.errorbar(k.rad, k.xc, yerr=k.er_xc, fmt='--o', mec='k', mew=1.2, color='skyblue', mfc='skyblue', capsize=3, label='Xc')
    ax6.errorbar(k.rad, k.yc, yerr=k.er_yc, fmt='--o', mec='k', mew=1.2, color='salmon', mfc='salmon', capsize=3, label='Yc')
    ax6.set_ylabel('$X_c, Y_c$ [arsces]', fontweight='bold')
    ax6.legend()
    if photo:
        ax6.set_xscale('log')
        ax6.set_xlabel('log)$_{10}$ R [arsces]', fontweight='bold')
    else:
        ax6.set_xlabel('R [arsces]', fontweight='bold')
    
    ax6.tick_params(axis='both', which='both', top=True, right=True)
    ax6.tick_params(axis='both', which='major', labelsize=10)
    ax6.yaxis.set_tick_params(length=6)
    ax6.xaxis.set_tick_params(width=2)
    ax6.yaxis.set_tick_params(width=2)
    ax6.xaxis.set_tick_params(length=6)
    ax5.tick_params(which='minor', length=3)
    ax6.tick_params(which='minor', width=1)
    ax6.tick_params(axis='both', which='both', top=True, right=True)
    for tick in ax6.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')  
    for tick in ax6.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax6.spines[axis].set_linewidth(2)
        
    fig.tight_layout()


#----------------------------------------------------------------------------
def vel_map_kinemetry():
    """
    This procedure ilustrates the simplest way of calling kinemetry to analyse a 
    velocity dispersion field, using the errors on the velocty,
    while also restricting the range of PA and Q
    
    If plot = False, there is no plotting for every R
    """


    file_dir = path.dirname(path.realpath(kin.__file__))
    file = file_dir + '/file/NGC6810_120_225.dat'
    xbin, ybin, velbin, er_velbin, sigbin, er_sigbin = np.genfromtxt(file, unpack=True)

    # call kinemetry
    start=time.time()
    k = kinemetry(xbin, ybin, velbin, x0=0.0, y0=0.0, scale=0.522, rangePA=[0,10], 
                  rangeQ=[0.4,0.99], ntrm=10, name='NGC6810', plot=False)
    end=time.time()
    print('finshed in ', end - start, ' seconds')
    
    #make radial plots
    plot_kinemetry_profiles_velocity(k, name='NGC6810 V-map')
    
    #make maps
    plot_kinemetry_maps(xbin, ybin, velbin, k)

#----------------------------------------------------------------------------
def vel_map_centre_kinemetry():
    """
    This procedure ilustrates a way of calling kinemetry to analyse a velocity 
    map, while having the centre coordinates as  variables
    
    If plot = False, there is no plotting for every R
    """

    file_dir = path.dirname(path.realpath(kin.__file__))
    file = file_dir + '/file/NGC6810_120_225.dat'
    xbin, ybin, velbin, er_velbin, sigbin, er_sigbin = np.genfromtxt(file, unpack=True)

    # call kinemetry
    start=time.time()
    k = kinemetry(xbin, ybin, velbin, fixcen=False, x0=0.0, y0=0.0, scale=0.522, 
                  ntrm=10, name='NGC6810', plot=False)
    end=time.time()
    print('finshed in ', end - start, ' seconds')
    
    #make radial plots
    plot_kinemetry_profiles_velocity(k, fitcentre=True, name='NGC6810 V-map, fit centre')
    
    #make maps
    plot_kinemetry_maps(xbin, ybin, velbin, k)





#----------------------------------------------------------------------------
def sig_map_kinemetry():

    """
    This procedure ilustrates the simplest way of calling kinemetry to analyse a 
    velocity dispersion field, using the errors on the velocty dispersion,
    while also restricting the range of PA and Q.
    
    If plot = False, there is no plotting for every R
    """

    file_dir = path.dirname(path.realpath(kin.__file__))
    file = file_dir + '/file/NGC6810_120_225.dat'
    xbin, ybin, velbin, er_velbin, sigbin, er_sigbin = np.genfromtxt(file, unpack=True)

    # call kinemetry
    start=time.time()
    k = kinemetry(xbin, ybin, sigbin, error=er_sigbin, even=True, ntrm=10, 
                  scale=0.522, rangePA=[0,10], rangeQ=[0.5,0.99], 
                  name='NGC6810', plot=False)
    end=time.time()
    print('finshed in ', end - start, ' seconds')
    
    #make radial plots
    plot_kinemetry_profiles_sigma(k, name='NGC6810, $\sigma$-map')
    
    #make maps
    plot_kinemetry_maps(xbin, ybin, sigbin, k, sigma=True)
    

#----------------------------------------------------------------------------
def sig_map_centre_kinemetry():

    """
    This procedure ilustrates the simplest way of calling kinemetry to analyse a 
    velocity dispersion field, using the errors on the velocty dispersion,
    while also restricting the range of PA and Q, AND fitting the centre
    
    If plot = False, there is no plotting for every R
    """

    file_dir = path.dirname(path.realpath(kin.__file__))
    file = file_dir + '/file/NGC6810_120_225.dat'
    xbin, ybin, velbin, er_velbin, sigbin, er_sigbin = np.genfromtxt(file, unpack=True)

    # call kinemetry
    start=time.time()
    k = kinemetry(xbin, ybin, sigbin, error=er_sigbin, even=True, ntrm=10, scale=0.522, 
              rangePA=[0,10], rangeQ=[0.5,0.99], x0=0.5, y0=0.5, fixcen=False,
              name='NGC6810', plot=False)
    end=time.time()
    print('finshed in ', end - start, ' seconds')
    
    #make radial plots
    plot_kinemetry_profiles_sigma(k, fitcentre=True, name='NGC6810, $\sigma$-map, fit centre')
    
    #make maps
    plot_kinemetry_maps(xbin, ybin, sigbin, k, sigma=True)
    



#----------------------------------------------------------------------------

# def vel_map_ellipse_kinemetry():
#     """
#     This procedure ilustrates a way of calling kinemetry to analyse a velocity 
#     map on predefined ellipses (e.g. isophotes). There is no plotting output.
#     """

#     file_dir = path.dirname(path.realpath(kin.__file__))
#     file = file_dir + '/file/NGC6810_120_225.dat'
#     xbin, ybin, velbin, er_velbin, sigbin, er_sigbin = np.genfromtxt(file, unpack=True)

#     rad=np.array([ 0.8       ,  1.68      ,  2.568     ,  3.4648    ,  4.37128   ,
#         5.288408  ,  6.2172488 ,  7.15897368,  8.11487105,  9.08635815,
#         10.07499397, 11.08249336, 12.1107427 , 13.16181697, 14.23799867,
#         15.34179854, 16.47597839, 17.64357623, 18.84793385, 20.09272724,
#         21.38199996, 22.72019996, 24.11221995, 25.56344195])
#     rad = rad/0.522 # move to pixel scale

#     pa = np.array([-85.25857204,  72.        ,  45.85858326,  41.29380557,
#         39.43680925,  38.64807806,  39.05834295,  39.87860798,
#         40.44709148,  40.08666996,  40.05975378,  40.7660266 ,
#         41.69108821,  42.3123077 ,  42.65269837,  42.92469078,
#         43.28886199,  43.57267972,  43.52283665,  43.48747184,
#         43.80463223,  43.81909029,  44.61293908,  45.08027104])

#     q = np.array([0.80611946, 0.96      , 0.87512803, 0.80406637, 0.7363923 ,
#         0.6903377 , 0.6702612 , 0.67309056, 0.68383574, 0.67566615,
#         0.64900005, 0.62983063, 0.62225559, 0.62215108, 0.62218298,
#         0.6165123 , 0.61042657, 0.60390107, 0.59806788, 0.59473859,
#         0.59608331, 0.60510225, 0.61687722, 0.62388233])
    
#     PAQ = np.zeros(2*pa.size)
#     for i in range(pa.size):
#         j=i*2
#         PAQ[j] = pa[i]
#         PAQ[j+1] =q[i]
        

#     # call kinemetry
#     start=time.time()
#     k = kinemetry(xbin, ybin, velbin, paq=PAQ, radius=rad, scale=0.522, name='NGC6810', plot=False)
#     end=time.time()
#     print('finshed in ', end - start, ' seconds')
    
#     #make radial plots
#     plot_kinemetry_profiles_velocity(k, fitcentre=True, name='NGC6810 V-map, fixed ellipses')
    
#     #make maps
#     plot_kinemetry_maps(xbin, ybin, velbin, k)
    
#----------------------------------------------------------------------------
if __name__ == '__main__':
    

    print('kinemetry on velocity map (fixed centre)')
    vel_map_kinemetry()
    
    print('kinemetry on velocity map + fit centre')
    vel_map_centre_kinemetry()
    

    print('kinemetry on velocity dispersion map, limited PA, Q, fixed centre')
    sig_map_kinemetry()

    print('kinemetry on velocity dispersion map, limited PA, Q')
    sig_map_centre_kinemetry()
    