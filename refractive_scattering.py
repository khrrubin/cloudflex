"""
Code for generating ray distributions, for FRB refractive scattering predictions
"""

import h5py
import numpy as np
import json
import numpy.fft
from math import *
import sys
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from unyt import \
    kpc, \
    pc, \
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
from trident_voigt import \
    solar_abundance, \
    tau_profile
from astropy.convolution import \
    Gaussian1DKernel, \
    convolve
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
import cmyt
import cmocean
import matplotlib.colors as colors
import os.path
import errno

from cloudflex import \
    Clouds, \
    generate_random_coordinates, \
    calculate_intersections, \
    calc_HI_frac

from IPython import embed

## change the default font to Avenir
plt.rcParams['font.family'] = 'Avenir'
# change the math font to Avenir
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Avenir'
plt.rcParams['mathtext.it'] = 'Avenir:italic'
plt.rcParams['mathtext.bf'] = 'Avenir:bold'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size=12)
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize


class RefractionPaths:
    """
    A class to represent a collection of paths as a series of arrays representing their
    positions, intersected cloudlet sizes, cloudlet pathlengths, etc.
    """
    def __init__(self, N=None, center=None, radius=None, coords=None, generate=True):
        """
        Can either accept ray coords or generate N random coords
        """
        self.N = N
        self.center = center
        self.radius = radius
        if generate:
            self.coords = generate_random_coordinates(self.N, self.center, self.radius)
            self.impact_par = np.sqrt(self.coords[:,0]**2 + self.coords[:,1]**2)
            self.index = np.arange(N) #for indexing spectra
            self.pathdict = {k: [] for k in range(N)}

        self.clouds = {}
        
        #self.n_clouds = np.zeros(N, dtype=float)
        #self.cloudlet_radii = {}
        #self.cloudlet_pathlengths = {}

        ## KHRR adding here -- assuming max number of cloudlets along sightline will always be < 1000
        #from IPython import embed
        #embed()
        #if(N):
        #    self.indiv_N = np.zeros((N,1000))
        #    self.indiv_bD = np.zeros((N,1000))
        #    self.indiv_dv = np.zeros((N,1000))


    def __repr__(self):
        return('RefractionPaths: %d paths centered on %s with radius %f' % (self.N, self.center, self.radius))

    def plot_refractionpaths(self, i, clouds, velocity=True, filename=None, ax=None):
        """
        Plot path's trajectory through clouds
        """
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(5.5, 1.5)
            standalone = True
        else:
            standalone = False

        # Method for using arbitrary colormap as color cycle
        n = len(self.clouds[i])
        colors = plt.cm.viridis(np.linspace(0,0.7,n))
        cloud_circs = [plt.Circle([clouds.centers[k,2], clouds.centers[k,1]], radius=clouds.radii[k], linewidth=0, color=colors[j]) for j, k in enumerate(self.clouds[i])]
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+g"))
        for k in range(len(cloud_circs)):
            cloud_circ = cloud_circs[k]
            ax.add_artist(cloud_circ)
            cloud_circ.set_alpha(0.7)
        ax.plot((-5, 5), (self.coords[i, 1], self.coords[i, 1]), color='k', alpha=0.7)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        if velocity:
            ax.quiver(clouds.centers[self.clouds[i],2], clouds.centers[self.clouds[i],1], clouds.velocities[self.clouds[i],2], clouds.velocities[self.clouds[i],1], width=0.005, color = colors)
        ax.set_xlim(-5,5)
        ax.set_ylim(self.coords[i, 1]-1, self.coords[i, 1]+1)
        ax.set_aspect(1)
        ax.set_xlabel('z (kpc)')
        ax.set_ylabel('y (kpc)')
        if standalone:
            plt.tight_layout()
            if filename is None:
                filename = 'ray_%04d.png' % i
            plt.savefig(filename)
            plt.close()

    def save(self, filename):
        """
        Saves to a JSON file
        """
        print("Saving refraction paths to %s" % filename)

        self.pathdict["npaths"] = self.N
        self.pathdict["center"] = self.center.tolist()
        self.pathdict["max_complex_impact_par"] = self.radius
        self.pathdict["electron_density"] = self.electron_density
        self.pathdict["coords"] = self.coords.tolist()

        with open(filename, 'wt') as fh:
            json.dump(self.pathdict, fh, indent=4)

        return

    def load(self, filename):
        """
        Loads from JSON file -- NEEDS UPDATING
        """
        print("Loading refraction paths from %s" % filename)
        with open(filename, 'rt') as fh:

            self.pathdict = json.load(fh) 
            self.N = self.pathdict["npaths"]
            self.center = self.pathdict["center"]
            self.radius = self.pathdict["max_complex_impact_par"]
            self.electron_density = self.pathdict["electron_density"]
            self.coords = self.pathdict["coords"]

        return



class PathGenerator():
    """

    Borrowed from cloudflex.SpectrumGenerator

    """
    def __init__(self, clouds):
        self.clouds = clouds
        p = clouds.params
        temperature_cloud = p['T_cl']
        self.metallicity_cloud = p['Z_cl']
        self.X_hydrogen = 0.74 # Hydrogen mass fraction
        self.Y_helium = 0.26    # Helium mass fraction -- ignoring metals.  Is this right????
        self.mean_molecular_weight = 1.0  ## IS THIS NEEDED??

        # Cloud properties (with units)
        self.temperature_cloud = temperature_cloud * K

        

    def make_path(self, refractionpaths, i):
        """
        For a path passing through the defined cloud distribution, compute
        cloud intersections
        
        """
        Z_cl = self.metallicity_cloud # units of Zsolar
        self.refractionpaths = refractionpaths
        self.i = i
        dls = calculate_intersections(refractionpaths.coords[i], self.clouds) * kpc
        mask = np.nonzero(dls)[0]
        self.mask = mask
        refractionpaths.clouds[i] = mask
        #refractionpaths.n_clouds[i] = len(mask)
        self.pathlengths = dls[mask]
        self.cloudlet_radii = self.clouds.radii[mask]  ## in kpc, no units here

        ipathdict = {}
        
        ipathdict["n_clouds"] = len(mask)

        if(len(mask) > 0):
            ipathdict["cloudlet_radii"] = (self.cloudlet_radii * kpc).to('pc').d.tolist()
            ipathdict["pathlength"] = self.pathlengths.to('pc').d.tolist()

        refractionpaths.pathdict[i] = ipathdict    
 
        
    def clear_path(self, total=False):
        """
        Clear out path to set up for another
        on another Ray
        """
        self.mask = []
        self.i = 0

        
    def generate_refractionpath_sample(self, N, center, radius):
        """
        Create a sample of N paths spanning an aperture with center and radius specified.
        Calculate their intersections with cloudlets, etc.
        Returns set of refraction paths
        """
        refractionpaths = RefractionPaths(N, center, radius, generate=True)

        # Determine HI ion abundance based on cool gas number density
        HI_frac = calc_HI_frac(self.clouds.params['n_cl'], X=self.X_hydrogen)

        print("n_cl = %f" % self.clouds.params['n_cl'])
        print("hI frac = %f" % HI_frac)

        # Calculate the hydrogen number density
        nH = self.X_hydrogen * self.clouds.params['n_cl']
        refractionpaths.electron_density = nH * HI_frac * (1.0 + (self.X_hydrogen/self.Y_helium))

        for i in tqdm(range(N), "Generating Refraction Paths"):
            self.make_path(refractionpaths, i)
            self.clear_path()
        return refractionpaths


if __name__ == '__main__':

    # Use this for testing

    fn = '/Users/krubin/Research/GPG/GIBLE/GIBLE_Complex_Suite/S98/MCLMIN/10/clouds_gb_18_mclmin_10.h5'

    clouds = Clouds()
    clouds.load(fn)
    params = clouds.params
    np.random.seed(seed=params['seed'])
   
    ## Generate and Plot Spectra for N random rays passing through domain
    ## Also plot distribution of clouds and rays

    # Create N random Rays passing through domain
    N = 10
    pg = PathGenerator(clouds)
    refractionpaths = pg.generate_refractionpath_sample(N, params['center'],  params['dclmax'])
    refractionpaths.save('test-rpths.json')

    rfps = RefractionPaths(generate=False)
    rfps.load('test-rpths.json')
    
    embed()
    