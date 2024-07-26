import h5py
import numpy as np
from math import *
import sys
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from unyt import \
    kpc, \
    cm, \
    K, \
    amu, \
    km, \
    s, \
    Msun, \
    angstrom, \
    c, \
    g, \
    proton_mass, \
    boltzmann_constant_cgs
import glob
from IPython import embed

from astropy.convolution import \
    Gaussian1DKernel, \
    convolve
from astropy.io import ascii
from astropy.table import Table
import matplotlib.ticker as ticker
from matplotlib.ticker import \
    MultipleLocator, \
    AutoMinorLocator
from matplotlib.collections import \
    PatchCollection
from matplotlib_scalebar.scalebar import \
    ScaleBar
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import cmyt
import cmocean
import matplotlib.colors as colors
import matplotlib as mpl
import os.path
import errno


# Load GIBLE clouds
filepath = '/Users/krubin/Research/GPG/GIBLE/Inputs/'

cat_list = glob.glob(filepath+'CloudCatalog_*RF512_z0.hdf5')

plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(6,5))
ax = plt.subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')
ax.set_xlabel(r'log Cloud Mass ($M_{\odot}$)')
ax.set_ylabel('Number of Clouds')

bins = np.arange(1,9,0.5)

for fil in cat_list:

    with h5py.File(fil, 'r') as f:
        gb_cloudPos = f["cloudPos"][:]            # cloud center of mass in kpc; z=0 plane aligned with galactic disk
        gb_cloudVel = f["cloudVel"][:]            # cloud 3D velocity in km/s
        gb_cloudVelDisp = f["cloudVelDisp"][:]    # cloud velocity dispersion
        gb_cloudMass = f["cloudMass"][:]          # total mass of cloud, solar masses
        gb_cloudNumCells = f["cloudNumCells"][:]  # number of cells in cloud
        gb_cloudMetal = f["cloudMetal"][:]           # EMPTY?? - mean metallicity of cloud in units of Zsun
        gb_cloudSize = f["cloudSize"][:]          # size computed as (3*volume / 4pi)^1/3 in kpc


    plt.hist(np.log10(gb_cloudMass), bins=bins, log=True, histtype='step')
    whresolved = gb_cloudMass > 10**5.5
    print(fil)
    print("Number of resolved clouds = ", len(gb_cloudMass[whresolved]))


filename = '/Users/krubin/Research/GPG/GIBLE/Inputs/CloudCatalog_S167RF4096_z0.hdf5'
with h5py.File(filename, 'r') as f:
    gb_cloudPos = f["cloudPos"][:]            # cloud center of mass in kpc; z=0 plane aligned with galactic disk
    gb_cloudVel = f["cloudVel"][:]            # cloud 3D velocity in km/s
    gb_cloudVelDisp = f["cloudVelDisp"][:]    # cloud velocity dispersion
    gb_cloudMass = f["cloudMass"][:]          # total mass of cloud, solar masses
    gb_cloudNumCells = f["cloudNumCells"][:]  # number of cells in cloud
    #gb_cloudMetal = f["cloudMetal"][:]           # EMPTY?? - mean metallicity of cloud in units of Zsun
    gb_cloudSize = f["cloudSize"][:]          # size computed as (3*volume / 4pi)^1/3 in kpc

plt.hist(np.log10(gb_cloudMass), bins=bins, log=True, histtype='step', color='k', lw=2)
whresolved = gb_cloudMass > 10**4.5
print("Number of resolved clouds for high res run = ", len(gb_cloudMass[whresolved]))
    

plt.tight_layout()
plt.savefig('gible_cloud_mass_function.pdf', format='pdf')


### Plot cumulative mass function
plt.figure(figsize=(6,5))
ax = plt.subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')
ax.set_xlabel(r'log Cloud Mass ($M_{\odot}$)')
ax.set_ylabel('Cumulative Mass Function')

bins = np.arange(1,10,0.5)

for fil in cat_list:

    with h5py.File(fil, 'r') as f:
        gb_cloudPos = f["cloudPos"][:]            # cloud center of mass in kpc; z=0 plane aligned with galactic disk
        gb_cloudVel = f["cloudVel"][:]            # cloud 3D velocity in km/s
        gb_cloudVelDisp = f["cloudVelDisp"][:]    # cloud velocity dispersion
        gb_cloudMass = f["cloudMass"][:]          # total mass of cloud, solar masses
        gb_cloudNumCells = f["cloudNumCells"][:]  # number of cells in cloud
        gb_cloudMetal = f["cloudMetal"][:]           # EMPTY?? - mean metallicity of cloud in units of Zsun
        gb_cloudSize = f["cloudSize"][:]          # size computed as (3*volume / 4pi)^1/3 in kpc


    ## Generate cumulative mass function
    mass_per_bin = np.zeros(len(bins)-1)
    for bb in range(len(bins)-1):
        wh = (np.log10(gb_cloudMass) > bins[bb]) & (np.log10(gb_cloudMass) < bins[bb+1])
        mass_per_bin[bb] = np.sum(gb_cloudMass[wh])

    cum_mass = np.flip(np.cumsum(np.flip(mass_per_bin)))

    #ax.hist(np.log10(gb_cloudMass), bins=bins, density=True, histtype='step', cumulative=-1)
    ax.stairs(cum_mass/cum_mass[0], edges=bins)
    whresolved = gb_cloudMass > 10**5
    print(fil)
    print("Number of resolved clouds = ", len(gb_cloudMass[whresolved]))


filename = '/Users/krubin/Research/GPG/GIBLE/Inputs/CloudCatalog_S167RF4096_z0.hdf5'
with h5py.File(filename, 'r') as f:
    gb_cloudPos = f["cloudPos"][:]            # cloud center of mass in kpc; z=0 plane aligned with galactic disk
    gb_cloudVel = f["cloudVel"][:]            # cloud 3D velocity in km/s
    gb_cloudVelDisp = f["cloudVelDisp"][:]    # cloud velocity dispersion
    gb_cloudMass = f["cloudMass"][:]          # total mass of cloud, solar masses
    gb_cloudNumCells = f["cloudNumCells"][:]  # number of cells in cloud
    #gb_cloudMetal = f["cloudMetal"][:]           # EMPTY?? - mean metallicity of cloud in units of Zsun
    gb_cloudSize = f["cloudSize"][:]          # size computed as (3*volume / 4pi)^1/3 in kpc
    

## Generate cumulative mass function
mass_per_bin = np.zeros(len(bins)-1)
for bb in range(len(bins)-1):
    wh = (np.log10(gb_cloudMass) > bins[bb]) & (np.log10(gb_cloudMass) < bins[bb+1])
    mass_per_bin[bb] = np.sum(gb_cloudMass[wh])

cum_mass = np.flip(np.cumsum(np.flip(mass_per_bin)))

#ax.hist(np.log10(gb_cloudMass), bins=bins, density=True, histtype='step', cumulative=-1)
#ax.stairs(cum_mass/cum_mass[0], edges=bins, lw=2, color='black')

#ax.hist(np.log10(gb_cloudMass), bins=bins, density=True, histtype='step', cumulative=-1)


whresolved = gb_cloudMass > 10**4.5
print("Number of resolved clouds for high res run = ", len(gb_cloudMass[whresolved]))
    

plt.tight_layout()
plt.savefig('gible_cloud_cum_mass_function.pdf', format='pdf')

embed()