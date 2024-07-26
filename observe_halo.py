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

from cloudflex import \
    generate_random_coordinates, \
    deposit_voigt, \
    calculate_intersections, \
    Rays, \
    Spectrum
from trident_voigt import \
    solar_abundance, \
    tau_profile


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

### PLAN ###
# 1. read in GIBLE cloud locations
# 2. place sightline
# 3. calculate intersection of GIBLE clouds, record:
#     masses, radial velocities, metallicities, velocity dispersions, coord of cloud center
#     --> not sure how to deal with edge of cloud -- maybe number density profile will help?
# 4. select appropriate rays, assemble list of columns, bD, 
# 5. calculate spectrum

class GIBLEClouds:
    """
    A class to represent a collection of clouds as a series of arrays representing their
    masses, positions, and velocities; Independent of Cloud class.  useful for saving to
    disk.
    """
    def __init__(self, centers=None, velocities=None, velocity_dispersions=None, masses=None, \
        number_of_cells=None, metallicities=None, radii=None):

        self.centers = centers
        self.velocities = velocities * km/s
        self.velocity_dispersions = velocity_dispersions * km/s
        self.masses = masses * Msun
        self.number_of_cells = number_of_cells
        self.metallicities = metallicities
        self.radii = radii * kpc
        self.index = np.arange(len(self.masses), dtype=int)
        

        
        # Calculate cloud number densities
        #gb_cloudMassDens = gb_cloudMass * Msun / ( (4/3) * np.pi * (gb_cloudSize*kpc)**3 )
        #gb_cloudMassDens = gb_cloudMassDens.to('kg/cm**3')
        #gb_cloudNumDens = gb_cloudMassDens / proton_mass    # is this right??
        mass_densities = self.masses / ( (4/3) * np.pi * (self.radii)**3 )
        mass_densities = mass_densities.to('kg/cm**3')
        self.number_densities = mass_densities / proton_mass  ### WHAT ABOUT MU????

        print('Reading in GIBLE cloud catalog')
        print('Minimum and maximum cloud z-velocities (km/s): %5.2f and %5.2f' % \
            (np.min(self.velocities[:,2]), np.max(self.velocities[:,2])))
        print('Minimum and maximum cloud masses (Msun): %.1g and %.1g' % \
            (np.min(self.masses.to('Msun')), np.max(self.masses.to('Msun'))))

    def plot_parameters_space(self):

        ###### Explore parameter space ###################

        Z_limits = 10.0**np.arange(-1.6,0.8,0.2)
        n_samp_per_cloudmass = np.zeros(len(Z_limits)-1)
        whloZ = Z_limits[:-1] < 0.09
        whmiZ = (Z_limits[:-1] > 0.09) & (Z_limits[:-1] < 3.98)
        whhiZ = Z_limits[:-1] > 3.98
        n_samp_per_cloudmass[whloZ] = 1.0
        n_samp_per_cloudmass[whmiZ] = 3.0
        n_samp_per_cloudmass[whhiZ] = 1.0

        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(9,7))

        for tt in range(len(Z_limits)-1):

            ax = plt.subplot(3,4,tt+1)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.5)
            ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
            ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax.set_xlim(1e2,1e8)
            ax.set_ylim(0.01,10)

            whZ = np.where((gible_clouds.metallicities > Z_limits[tt]) & (gible_clouds.metallicities < Z_limits[tt+1]))[0]
            ax.plot(gible_clouds.masses.to('Msun')[whZ], gible_clouds.radii[whZ], linestyle='none', marker='o', color='k', ms=3, alpha=0.5)
            #ax.set_xlabel(r'Cloud Mass ($M_{\odot}$)')
            #ax.set_ylabel(r'Cloud Radius (kpc)')
            ax.set_xscale('log')
            ax.set_yscale('log')

            cloud_mass_array = np.logspace(2,7,11)
            cloud_rad_lower = 10.0**(0.35*(np.log10(cloud_mass_array)-2.0) - 1.6)
            cloud_rad_upper = 10.0**(0.3*(np.log10(cloud_mass_array)-2.0) - 0.6)

            cloud_rad_lower_hiZ = 10.0**(0.35*(np.log10(cloud_mass_array)-2.0) - 1.1)
            cloud_rad_upper_hiZ = 10.0**(0.3*(np.log10(cloud_mass_array)-2.0) - 0.65)

            cloud_rad_lower_loZ = 10.0**(0.35*(np.log10(cloud_mass_array)-2.0) - 1.2)
            cloud_rad_upper_loZ = 10.0**(0.3*(np.log10(cloud_mass_array)-2.0) - 0.85)

            if((Z_limits[tt] > 0.09) & (Z_limits[tt] < 3.98)):
                ax.plot(cloud_mass_array, cloud_rad_lower, color='orange')
                ax.plot(cloud_mass_array, cloud_rad_upper, color='orange')

            elif(Z_limits[tt] < 0.09):
                ax.plot(cloud_mass_array, cloud_rad_lower_loZ, color='red')
                ax.plot(cloud_mass_array, cloud_rad_upper_loZ, color='red')

            elif(Z_limits[tt] > 3.98):
                ax.plot(cloud_mass_array, cloud_rad_lower_hiZ, color='cyan')
                ax.plot(cloud_mass_array, cloud_rad_upper_hiZ, color='cyan')

            if(tt%4==0):
                ax.set_ylabel(r'Cloud Radius (kpc)')
            if(tt>7):
                ax.set_xlabel(r'Cloud Mass ($M_{\odot}$)')
        
            label_str = r'[{0:.2f}, {1:.2f}]'.format(Z_limits[tt],Z_limits[tt+1])
            ax.text(0.05, 0.85, label_str, transform=ax.transAxes, ha='left')

           
        plt.tight_layout()
        plt.savefig('gible_cloud_mass_vs_radius_S98.pdf', format='pdf')


        ### Refining sampling ###############

        Z_limits = 10.0**np.arange(-2.5,1.0,0.5)
        ndens_limits = np.logspace(-6,2,9)
        n_samp_per_cloudmass = np.zeros(len(ndens_limits)-1)
        whn5 = (ndens_limits[:-1] < 0.0001) & (ndens_limits[:-1] > 1.0e-6)
        whn4 = (ndens_limits[:-1] > 0.00001) & (ndens_limits[:-1] < 0.001)
        whn3 = (ndens_limits[:-1] > 0.0001) & (ndens_limits[:-1] < 0.01)
        whn2 = (ndens_limits[:-1] > 0.001) & (ndens_limits[:-1] < 0.1)
        n_samp_per_cloudmass[whn5] = 3
        n_samp_per_cloudmass[whn4] = 5
        n_samp_per_cloudmass[whn3] = 5
        n_samp_per_cloudmass[whn2] = 3
        n_samp_per_cloudmass = np.int64(n_samp_per_cloudmass)
        maxn_radii = int(np.max(n_samp_per_cloudmass))

        cloud_mass_array = np.logspace(3,8,10)
        cloud_Z_array = 10.0**np.array([(np.log10(Z_limits[i])+np.log10(Z_limits[i+1]))/2 for i in range(len(Z_limits)-1)])
        cloud_ndens_array = 10.0**np.array([(np.log10(ndens_limits[i])+np.log10(ndens_limits[i+1]))/2 for i in range(len(ndens_limits)-1)])
        cloud_rad_mass_ndens_Z_grid = np.zeros((len(cloud_mass_array), len(cloud_Z_array), len(cloud_ndens_array), maxn_radii))-999.0

        cloud_rad_lower_ndens_n5 = 10.0**(0.35*(np.log10(cloud_mass_array)-2.0) - 0.75)
        cloud_rad_upper_ndens_n5 = 10.0**(0.3*(np.log10(cloud_mass_array)-2.0) - 0.3)

        cloud_rad_lower_ndens_n4= 10.0**(0.35*(np.log10(cloud_mass_array)-2.0) - 1.2)
        cloud_rad_upper_ndens_n4 = 10.0**(0.33*(np.log10(cloud_mass_array)-2.0) - 0.55)

        cloud_rad_lower_ndens_n3 = 10.0**(0.35*(np.log10(cloud_mass_array)-2.0) - 1.4)
        cloud_rad_upper_ndens_n3 = 10.0**(0.3*(np.log10(cloud_mass_array)-2.0) - 0.8)

        cloud_rad_lower_ndens_n2 = 10.0**(0.35*(np.log10(cloud_mass_array)-2.0) - 1.6)
        cloud_rad_upper_ndens_n2 = 10.0**(0.3*(np.log10(cloud_mass_array)-2.0) - 1.2)


        for m in range(len(cloud_mass_array)):
            for n in range(len(cloud_ndens_array)):

                if (cloud_ndens_array[n] < 0.0001):

                    rad_lo = cloud_rad_lower_ndens_n5
                    rad_hi = cloud_rad_upper_ndens_n5
                    flg_Z = np.array([False,False,False,False,True,True])

                elif ((cloud_ndens_array[n] > 0.0001) & (cloud_ndens_array[n] < 0.001)):

                    rad_lo = cloud_rad_lower_ndens_n4
                    rad_hi = cloud_rad_upper_ndens_n4
                    flg_Z = np.array([False,True,True,True,True,True])


                elif ((cloud_ndens_array[n] > 0.001) & (cloud_ndens_array[n] < 0.01)):

                    rad_lo = cloud_rad_lower_ndens_n3
                    rad_hi = cloud_rad_upper_ndens_n3
                    flg_Z = np.array([True, True, True, True, True, True])


                else:

                    rad_lo = cloud_rad_lower_ndens_n2
                    rad_hi = cloud_rad_upper_ndens_n2
                    flg_Z = np.array([False,False,False,True,True,False])


                #if((n_samp_per_cloudmass[z]>0) & (n_samp_per_cloudmass[z] <= 1)):
                #    rad_samp = 10.0**np.linspace(np.log10(rad_lo[m]), np.log10(rad_hi[m]), n_samp_per_cloudmass[z]+2)
                #    cloud_rad_mass_ndens_Z_grid[m,z,:n_samp_per_cloudmass[z]] = rad_samp[1:-1]

                
                rad_samp = 10.0**np.linspace(np.log10(rad_lo[m]), np.log10(rad_hi[m]), n_samp_per_cloudmass[n])
                for z in range(len(flg_Z)):
                    if flg_Z[z]==True:
                        cloud_rad_mass_ndens_Z_grid[m,z,n,:n_samp_per_cloudmass[n]] = rad_samp
        
        #whrad = np.where(cloud_rad_mass_Z_grid > 0.0)
        #print("Number of grid points = ", len(cloud_rad_mass_Z_grid[whrad]))

        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(16,12))

        for tt in range(len(Z_limits)-1):

            for nn in range(len(ndens_limits)-1):

                ax = plt.subplot(6,8,(nn+1) + 8*(tt))
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(1.5)
                ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
                ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                ax.set_xlim(1e2,1e9)
                ax.set_ylim(0.01,20)

                whZ = np.where((gible_clouds.metallicities > Z_limits[tt]) & (gible_clouds.metallicities < Z_limits[tt+1]) \
                & (gible_clouds.number_densities > ndens_limits[nn]) & (gible_clouds.number_densities < ndens_limits[nn+1]))[0]
                ax.plot(gible_clouds.masses.to('Msun')[whZ], gible_clouds.radii[whZ], linestyle='none', marker='o', color='k', ms=3, alpha=0.5)
                #ax.set_xlabel(r'Cloud Mass ($M_{\odot}$)')
                #ax.set_ylabel(r'Cloud Radius (kpc)')
                ax.set_xscale('log')
                ax.set_yscale('log')

                if (ndens_limits[nn] < 0.0001):

                    ax.plot(cloud_mass_array, cloud_rad_lower_ndens_n5, color='red')
                    ax.plot(cloud_mass_array, cloud_rad_upper_ndens_n5, color='red')
                    
                elif ((ndens_limits[nn] >= 0.0001) & (ndens_limits[nn] < 0.001)):

                    ax.plot(cloud_mass_array, cloud_rad_lower_ndens_n4, color='orange')
                    ax.plot(cloud_mass_array, cloud_rad_upper_ndens_n4, color='orange')
                    
                elif ((ndens_limits[nn] >= 0.001) & (ndens_limits[nn] < 0.01)):

                    ax.plot(cloud_mass_array, cloud_rad_lower_ndens_n3, color='green')
                    ax.plot(cloud_mass_array, cloud_rad_upper_ndens_n3, color='green')
                
                elif ((ndens_limits[nn] >= 0.01) & (ndens_limits[nn] < 0.1)):

                    ax.plot(cloud_mass_array, cloud_rad_lower_ndens_n2, color='cyan')
                    ax.plot(cloud_mass_array, cloud_rad_upper_ndens_n2, color='cyan')
                
                

                #if((Z_limits[tt] > 0.09) & (Z_limits[tt] < 3.1)):
                #    ax.plot(cloud_mass_array, cloud_rad_lower, color='orange')
                #    ax.plot(cloud_mass_array, cloud_rad_upper, color='orange')

                #elif(Z_limits[tt] < 0.09):
                #    ax.plot(cloud_mass_array, cloud_rad_lower_loZ, color='red')
                #    ax.plot(cloud_mass_array, cloud_rad_upper_loZ, color='red')

                #elif(Z_limits[tt] > 3.1):
                #    ax.plot(cloud_mass_array, cloud_rad_lower_hiZ, color='cyan')
                #    ax.plot(cloud_mass_array, cloud_rad_upper_hiZ, color='cyan')

                # Show sampling!
                whgrid = (cloud_ndens_array > ndens_limits[nn]) & (cloud_ndens_array < ndens_limits[nn+1]) 
                whZ = (cloud_Z_array > Z_limits[tt]) & (cloud_Z_array < Z_limits[tt+1])
                #for zz in range(len(cloud_rad_mass_ndens_Z_grid[0,:,0,0])):
                for rr in range(len(cloud_rad_mass_ndens_Z_grid[0,0,0,:])):
                    ax.plot(cloud_mass_array, cloud_rad_mass_ndens_Z_grid[:,whZ,whgrid,rr], ls='None', marker='o', color='red', alpha=0.5)
        
                #if(tt%4==0):
                    #ax.set_ylabel(r'Cloud Radius (kpc)')
                #if(tt>2):
                    #ax.set_xlabel(r'Cloud Mass ($M_{\odot}$)')
        
                Zlabel_str = r'[{0:.2f}, {1:.2f}]'.format(Z_limits[tt],Z_limits[tt+1])
                ax.text(0.05, 0.85, Zlabel_str, transform=ax.transAxes, ha='left')
                nlabel_str = r'[{0:.2g}, {1:.2g}]'.format(ndens_limits[nn],ndens_limits[nn+1])
                ax.text(0.05, 0.05, nlabel_str, transform=ax.transAxes, ha='left')

           
        plt.tight_layout()
        plt.savefig('gible_cloud_mass_vs_radius_coarseZ_S98.pdf', format='pdf')

        embed()

    def plot_clouds(self):

        # Plot cloud locations
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6.25)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
        ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')

        ax.set_xlim(-230,230)
        ax.set_ylim(-230,230)
        ax.set_aspect(1)

        ax.set_xlabel('x (kpc)')
        ax.set_ylabel('y (kpc)')

        # size unit is 1/72 inch
        imsize = 8.0  # inches -- NEED TO MAKE SURE THIS IS REALLY RIGHT
        fractional_cloudsize = self.radii.value / 460.0
        fractional_pt_cloudsize = fractional_cloudsize * imsize * 72.0 #* np.sqrt(np.pi)

        cm = mpl.cm.get_cmap('coolwarm')
        pos = ax.scatter(self.centers[:,0],self.centers[:,1],s=fractional_pt_cloudsize**2,c=self.velocities.value[:,2],cmap=cm)

        
        n_clouds = len(self.radii.value)
        N = 1000
        n_batch = int(np.floor(n_clouds / N))
        #for j in tqdm(range(n_batch), "Plotting GIBLE Clouds"):

            #X,Y = np.meshgrid(self.radii.value[j*N+i],self.radii.value[j*N+i])
            #XY = np.column_stack((self.centers[j*N+i,0], self.centers[j*N+i,1]))

            #ec = EllipseCollection(X, Y, X, units='x', offsets=XY, offset_transform=ax.transData)
            #ec.set_array(self.velocities[j*N+i,2].value)
            #ax.add_collection(ec)
            #ax.autoscale_view()

        
            #circles = [plt.Circle((self.centers[j*N+i,0], self.centers[j*N+i,1]), radius=self.radii.value[j*N+i]) for i in range(N)]
            #vel_for_cb = [self.velocities.value[j*N+i,2] for i in range(N)]
            #collection = PatchCollection(circles, cmap='coolwarm')
            #collection.set_array(vel_for_cb)
            #collection.set_alpha(0.5)
        #    #collection.set_alpha(alphas[j*N:(j+1)*N])
        #    #collection.set_color('tab:blue')
            #collection.set_linewidth(0)
            #ax.add_collection(collection)
        #    # Plot remainder
            #remainder = n_clouds % N
            #circles = [plt.Circle((self.centers[i,0], self.centers[i,1]), radius=self.radii.value[i]) for i in range(N*n_batch,n_clouds)]
            #vel_for_cb = [self.velocities.value[i,2] for i in range(N*n_batch,n_clouds)]
            #collection = PatchCollection(circles, cmap='coolwarm')
            #collection.set_alpha(0.5)
        #    #collection.set_alpha(alphas[n_batch*N:])
        #    #collection.set_color('tab:blue')
            #collection.set_linewidth(0)
            #ax.add_collection(collection)

        fig.colorbar(pos, ax=ax, label='Radial Velocity (km/s)')
        #cbar.set_label('Radial Velocity (km/s)')
       
        plt.tight_layout()
        plt.savefig("gible_cloudmap_velocity.png")
        plt.close("all")


class GIBLERays:
    """
    A class to represent a collection of rays as a series of arrays representing their
    positions, EWs, column densities, etc.
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
            self.index = np.arange(N) #for indexing spectra

        self.gible_clouds = {}
        self.column_densities = np.zeros(N, dtype=float)
        self.n_gible_clouds = np.zeros(N, dtype=float)
        self.n_all_clouds = np.zeros(N, dtype=float)
        self.EWs = np.zeros(N, dtype=float)
        self.spectra = {}

        ## KHRR adding here -- assuming max number of cloudlets along sightline will always be < 100
        #from IPython import embed
        #embed()
        if(N):
            self.indiv_N = np.zeros((N,300))
            self.indiv_bD = np.zeros((N,300))
            self.indiv_dv = np.zeros((N,300))

    def save(self, filename):
        """
        Saves to an HDF5 file
        """
        print("Saving rays to %s" % filename)

        with h5py.File(filename, 'w') as f:
            f.create_dataset("coords", data=self.coords)
            f.create_dataset("column_densities", data=self.column_densities)
            f.create_dataset("EWs", data=self.EWs)
            f.create_dataset("n_gible_clouds", data=self.n_gible_clouds)
            f.create_dataset("n_all_clouds", data=self.n_all_clouds)
            f.create_dataset("indiv_N", data=self.indiv_N)
            f.create_dataset("indiv_bD", data=self.indiv_bD)
            f.create_dataset("indiv_dv", data=self.indiv_dv)
            f.attrs['N'] = self.N
            f.attrs['center'] = self.center
            f.attrs['radius'] = self.radius
        
            # TODO: Save attached spectra dictionary as a group with individual
            # spectra as datasets
            f.create_dataset("lambda_field", data=self.spectra[0].lambda_field)
            grp_spectra = f.create_group("spectra")

            for i in range(len(gible_rays.spectra)):
                spec = grp_spectra.create_dataset(f'spec_{i:05}', data=self.spectra[i].flux_field)

        return

    def load(self, filename):
        """
        Loads from an HDF5 file
        """
        print("Loading rays from %s" % filename)
        with h5py.File(filename, 'r') as f:
            self.coords = f["coords"][:]
            self.column_densities = f["column_densities"][:]
            self.EWs = f["EWs"][:]
            self.n_gible_clouds = f["n_gible_clouds"][:]
            self.n_all_clouds = f["n_all_clouds"][:]
            self.indiv_N = f["indiv_N"][:]
            self.indiv_bD = f["indiv_bD"][:]
            self.indiv_dv = f["indiv_dv"][:]
            self.N = f.attrs['N']
            self.center = f.attrs['center']
            self.radius = f.attrs['radius']

            if("lambda_field" in f):
                self.lambda_field = f["lambda_field"][:]

                flux_fields = {}
                for i in range(len(f["spectra"].keys())):
                    flux_fields[i] = f["spectra"][f'spec_{i:05}'][:]
                self.flux_fields = flux_fields
        return


    def print_ascii(self, filename, updir='RaysAscii'):
        """
        Generate text files of spectra to prepare for MC-ALF fitting
        """
        print("Loading rays from %s" % filename)
        with h5py.File(filename, 'r') as f:
            self.coords = f["coords"][:]
            self.column_densities = f["column_densities"][:]
            self.EWs = f["EWs"][:]
            self.n_gible_clouds = f["n_gible_clouds"][:]
            self.n_all_clouds = f["n_all_clouds"][:]
            self.indiv_N = f["indiv_N"][:]
            self.indiv_bD = f["indiv_bD"][:]
            self.indiv_dv = f["indiv_dv"][:]
            self.N = f.attrs['N']
            self.center = f.attrs['center']
            self.radius = f.attrs['radius']

            if("lambda_field" in f):
                self.lambda_field = f["lambda_field"][:]

                uppath = os.path.join('.', updir)
                if not os.path.exists(uppath):
                    os.makedirs(uppath)
                os.chdir(uppath)

                print("Printing ascii spectra...")

                flux_fields = {}
                for i in range(len(f["spectra"].keys())):

                    flux_fields[i] = f["spectra"][f'spec_{i:05}'][:]

                    if(self.n_all_clouds[i] > 0):
                        data = Table()
                        data['wave'] = self.lambda_field
                        data['flux'] = flux_fields[i]
                        data['err'] = 0.01 + np.zeros(len(flux_fields[i]))

                        ascii.write(data, f'spec_{i:05}.txt', overwrite=True, \
                            formats={'wave': '%12.5f', 'flux': '%12.9f', 'err': '%12.9f'})

                self.flux_fields = flux_fields
                os.chdir('..')

                
            else:
                print("No spectra saved with these rays")
        return


class GIBLESpectrumGenerator():
    """
    Very similar to cloudflex.SpectrumGenerator

    Sets spectrum as centered around Mg II 2796 line
    """
    def __init__(self, gible_clouds, show=True, complex_path=None, complex_mtots=None, complex_dclmaxs=None, \
                    gible_only=True, instr_pixel_width=None, instr_resolution=45000., debug=False):

        #self.instrument = 'HIRES'
        self.line = 'Mg II'
        self.gible_clouds = gible_clouds
        self.complex_path = complex_path
        self.gible_only = gible_only
        self.complex_mtots = complex_mtots
        self.complex_dclmaxs = complex_dclmaxs

        # Mg II Line Info
        self.lambda_0 = 2796.35 #* angstrom #AA
        self.f_value = 0.6155
        self.gamma = 2.625e+08 #/ s # 1 / s>
        self.mg_mass = 24.305 * amu
        self.show = True
        self.debug = debug

        if(gible_only):
            temperature_cloud = 1.0e4
            self.metallicity_cloud = 0.33  ### NEED TO ADJUST THIS

            # Cloud properties (with units)
            self.temperature_cloud = temperature_cloud * K

        # Define spectral pixel bin width by velocity
        #self.bin_width = 1e-2 # in angstroms
        # HIRES bin widths = 2.2 km/s ~ 0.02 Angstroms for Mg II
        # Use HIRES bin widths / 2 for smoothness in plotting individual cloudlet spectra,
        # then for EW calc and plotting full spectrum, downgrade resolution to HIRES
        if(instr_pixel_width):
            velocity_bin_width = instr_pixel_width * km/s
        else:
            # default to HIRES
            velocity_bin_width = 2.2 * km/s
        z = velocity_bin_width / c
        bin_width = (z * self.lambda_0).d # in angstroms
        self.bin_width = bin_width / 2. # 2x finer than HIRES
        self.instr_resolution = instr_resolution

        # Spectrum consists of lambda, tau, velocity, and flux fields
        # Flux is derived from tau (optical depth) and
        # velocity is derived from lambda (wavelength)

        self.lambda_range = 8 # in angstroms
        self.lambda_min = self.lambda_0 - self.lambda_range / 2. # in angstroms
        self.lambda_max = self.lambda_0 + self.lambda_range / 2. # in angstroms
        self.lambda_field = np.linspace(self.lambda_min, self.lambda_max, \
                                        int(self.lambda_range / self.bin_width))
        z = (self.lambda_field / self.lambda_0) - 1
        self.velocity_field = z * c
        self.velocity_field.convert_to_units('km/s')
        self.clear_spectrum()
        self.create_LSF_kernel()

        # Initialize these for averaging spectrum over multiple sightlines
        self.tau_sum_field = np.zeros_like(self.tau_field)
        self.flux_sum_field = np.zeros_like(self.tau_field)
        self.n_sightlines = 0
        self.sum_intersections = 0
        self.sum_column_density = 0

    def make_spectrum(self, gible_rays, i, mgII_frac, attach=False):
        """
        For a ray passing through the defined cloud distribution, Actually make the
        spectrum by stepping through and depositing voigt profiles for each cloud
        intersected by ray.

        Optionally attach each spectrum to the Rays class that called it
        """
      
        self.gible_rays = gible_rays
        self.i = i
        #dls = calculate_intersections(gible_rays.coords[i], self.gible_clouds, GIBLE=True) * kpc
        impact_par = np.linalg.norm(gible_rays.coords[i] - self.gible_clouds.centers[:, :2], axis=1) * kpc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pathlengths = 2*(self.gible_clouds.radii.value**2 - impact_par.value**2)**0.5
        dls = np.nan_to_num(pathlengths) * kpc

        mask = np.nonzero(dls)[0]
        self.mask = mask
        gible_rays.gible_clouds[i] = mask
        gible_rays.n_gible_clouds[i] = len(mask)

        # Storage for assembly of absorbers
        all_dls = []
        all_mgII_column_densities = []
        all_radial_velocities = []
        all_broadening = []
        n_all_clouds = 0

        if not (self.gible_only):
            
            for index, j in enumerate(mask):

                # Find complex of closest total mass
                imtot = np.argmin(np.abs(self.gible_clouds.masses.to('Msun').value[j] - self.complex_mtots))

                # Find complex of closest size
                idclmax = np.argmin(np.abs(self.gible_clouds.radii.value[j] - np.array(self.complex_dclmaxs[imtot])))

                # Read in corresponding set of rays
                #mtot_path = '%.1g' % self.complex_mtots[imtot]
                path = '%.1g_%.3g' % (self.complex_mtots[imtot], self.complex_dclmaxs[imtot][idclmax])
                ray_fil = 'rays_mtot_%.1g_dclmax_%.3g.h5' % (self.complex_mtots[imtot], self.complex_dclmaxs[imtot][idclmax])
                
                rays = Rays(generate=False)
                rays.load(os.path.join(self.complex_path, path, ray_fil))
                rays_impact_par = np.sqrt(rays.coords[:,0]**2 + rays.coords[:,1]**2)
                

                #if(impact_par[j] > 5.0*kpc):
                #    wh_impact = rays_impact_par > 4.0
                #else:
                # This can technically be negative -- clean this up
                #wh_impact = (rays_impact_par > 0.9*impact_par.value[j]) & \
                #        (rays_impact_par < 1.1*impact_par.value[j])

                # Now draw ray at random from rays[wh_impact]
                #n_impact_par = len(rays.EWs[wh_impact])
                n_impact_par = len(rays.EWs)
                irandom = np.random.randint(0,high=n_impact_par,size=None)
                
                for k in range(int(rays.n_clouds[irandom])):
                    all_mgII_column_densities.append(rays.indiv_N[irandom,k])
                    all_broadening.append(rays.indiv_bD[irandom,k])

                    # Is this correct????
                    all_radial_velocities.append(rays.indiv_dv[irandom,k] + self.gible_clouds.velocities.value[j,2]) 
                    n_all_clouds = n_all_clouds + 1
                

            all_mgII_column_densities = np.array(all_mgII_column_densities) / cm**2
            all_radial_velocities = np.array(all_radial_velocities) * km/s
            all_broadening = np.array(all_broadening) * km/s
            all_broadening.convert_to_units('cm/s')

            gible_rays.column_densities[i] =  np.sum(all_mgII_column_densities)
            gible_rays.n_all_clouds[i] = n_all_clouds
            self.n_all_clouds = n_all_clouds

            self.tau_field = np.zeros_like(self.lambda_field)
            self.tau_fields = np.zeros([len(all_mgII_column_densities), self.lambda_field.shape[0]])
        
        else:
                    
            column_densities = dls * self.gible_clouds.number_densities
            Z_cl = self.metallicity_cloud # units of Zsolar
            X_hydrogen = 0.74 # Hydrogen mass fraction
            mg_column_densities = solar_abundance['Mg'] * column_densities * Z_cl * X_hydrogen
            mgII_column_densities = mg_column_densities * mgII_frac
            all_mgII_column_densities = mgII_column_densities[mask]
            gible_rays.column_densities[i] = np.sum(all_mgII_column_densities)
            all_radial_velocities = self.gible_clouds.velocities[mask,2]
        
            self.tau_field = np.zeros_like(self.lambda_field)
            self.tau_fields = np.zeros([len(mask), self.lambda_field.shape[0]])

            # Thermal broadening b parameter
            self.thermal_b = np.sqrt((2 * boltzmann_constant_cgs * self.temperature_cloud) /
                                self.mg_mass)
            self.thermal_b.convert_to_units('cm/s')

            vel_disp = self.gible_clouds.velocity_dispersions.to('cm/s')
            all_broadening = np.sqrt(self.thermal_b**2 + (vel_disp[mask])**2)

            gible_rays.n_all_clouds[i] = gible_rays.n_gible_clouds[i]
            self.n_all_clouds = gible_rays.n_gible_clouds[i]

            ## PROBABLY NEED TO CONVERT VEL DISP TO BD

        if(self.debug):
            if(len(mask)>0):
                print("Component info: ", i)
                print("MgII column densities: ", all_mgII_column_densities)
                print("Velocities: ", all_radial_velocities)
                print("Broadening: ", all_broadening)
                #print("Cloud masses: ", self.clouds.masses[mask])

        self.column_density_list = np.zeros(len(all_mgII_column_densities))
        self.dv_list = np.zeros(len(all_mgII_column_densities))
        self.bD_list = np.zeros(len(all_mgII_column_densities))

        #for index, j in enumerate(mask):
        for index in range(len(all_mgII_column_densities)):
            self.tau_fields[index,:] += deposit_voigt(all_radial_velocities.value[index],
                                        all_mgII_column_densities[index], \
                                        self.lambda_0, self.f_value, self.gamma, \
                                        all_broadening[index], self.lambda_field)
            self.column_density_list[index] = all_mgII_column_densities[index]
            self.dv_list[index] = all_radial_velocities[index]
            self.bD_list[index] = all_broadening.to('km/s')[index]

        ## KHRR adding here
        gible_rays.indiv_N[i,:len(all_mgII_column_densities)] = self.column_density_list
        gible_rays.indiv_bD[i,:len(all_mgII_column_densities)] = self.bD_list
        gible_rays.indiv_dv[i,:len(all_mgII_column_densities)] = self.dv_list

        self.tau_field = np.sum(self.tau_fields, axis=0)
        self.flux_fields = np.exp(-self.tau_fields)
        self.flux_field = np.exp(-self.tau_field)

        # keep track of how many spectra are being averaged together
        # KHRR -- NOT SURE IF THIS IS RIGHT
        self.n_sightlines += 1
        self.flux_sum_field += self.flux_field
        self.sum_intersections += gible_rays.n_all_clouds[i]
        self.sum_column_density += gible_rays.column_densities[i]

        # if the ray intersected clouds, then apply LSF and calculate EW
        if gible_rays.n_all_clouds[i] > 0:
            self.apply_LSF()
            self.calculate_equivalent_width()
            gible_rays.EWs[i] = self.EW

        # Attach the spectra to the Rays object
        if attach:
            self.gible_rays.spectra[i] = Spectrum(self.lambda_field, self.flux_field, \
                                            self.lambda_0, i, self.EW, \
                                            gible_rays.column_densities[i])

    def make_mean_spectrum(self):
        """
        Create the flux mean field as the arithmetic mean of all flux fields
        across all rays
        """
        self.flux_mean_field = self.flux_sum_field / self.n_sightlines
        self.apply_LSF(mean=True)
        self.calculate_equivalent_width(mean=True)

    def clear_spectrum(self, total=False):
        """
        Clear out flux fields for a given spectrum to re-use SpectrumGenerator
        on another Ray
        """
        self.ray = None
        self.tau_field = np.zeros_like(self.lambda_field)
        self.flux_field = np.zeros_like(self.lambda_field)
        self.EW = 0.
        self.mask = []
        self.i = 0

        if total:
            self.n_sightlines = 0
            self.sum_intersections = 0
            self.sum_column_density = 0
            self.gible_rays = None
            self.i = 0
            self.mask = []
            self.tau_sum_field = 0
            self.flux_sum_field = 0
            self.flux_mean_field = 0
            self.n_sightlines = 0
            self.sum_intersections = 0
            self.sum_column_density = 0

    def calculate_equivalent_width(self, mean=False):
        """
        Calculate equivalent width of an absorption spectrum in angstroms
        """
        # degrading resolution of spectrum to HIRES when calculating EW
        if mean:
            flux_field = self.flux_mean_field[::2]
            self.mean_EW = np.sum(np.ones_like(flux_field) - flux_field) * self.bin_width*2 * angstrom
        else:
            flux_field = self.flux_field[::2]
            self.EW = np.sum(np.ones_like(flux_field) - flux_field) * self.bin_width*2 * angstrom

    def create_LSF_kernel(self):     #, resolution=45000.):
        """
        Create line spread function kernel to smear out spectra to appropriate
        spectral resolution.  Defaults to Keck HIRES R=45000.
        """
        delta = self.lambda_0 / self.instr_resolution
        # Note: delta is resolution ~ FWHM
        # whereas Gaussian1DKernel gives width in 1 stddev
        # FWHM ~ 2.35sigma
        # HIRES FWHM Res ~ 0.062 Angstroms -> 7 km/s

        gaussian_width = delta / (self.bin_width * 2.35)
        self.LSF_kernel = Gaussian1DKernel(gaussian_width)

    def apply_LSF(self, mean=False):
        """
        Apply the line spread function after all voigt profiles
        have been deposited.
        """

        if mean:
            self.flux_mean_field = convolve(self.flux_mean_field, \
                                            self.LSF_kernel, boundary='extend')
            np.clip(self.flux_mean_field, 0, np.inf, out=self.flux_mean_field)
        else:
            for j in range(int(self.n_all_clouds)):
                self.flux_fields[j,:] = convolve(self.flux_fields[j,:], self.LSF_kernel, boundary='extend')
            np.clip(self.flux_field, 0, np.inf, out=self.flux_field)
            self.flux_field = convolve(self.flux_field, self.LSF_kernel, boundary='extend')
            np.clip(self.flux_field, 0, np.inf, out=self.flux_field)


    def plot_spectrum(self, filename=None, component=False,
                      velocity=True, annotate=True, mean=False, cloud_vels=False,
                      ray_plot=False):
        """
        Plot spectrum in angstroms or km/s to filename
        """
        #if ray_plot:
        #    fig, axs = plt.subplots(2, 1, height_ratios=[3,1])
        #    ax = axs[0]
        #    fig.set_size_inches(5.5, 6)
        #    self.rays.plot_ray(self.i, clouds=self.clouds, ax=axs[1])
        #    j = k = 0
        #else:
        fig, ax = plt.subplots()
        fig.set_size_inches(5.5, 4.5)
        if mean:
            flux_field = self.flux_mean_field
            EW_val = self.mean_EW
            column_density = self.sum_column_density
            n_intersections = self.sum_intersections
        else:
            flux_field = self.flux_field
            EW_val = self.gible_rays.EWs[self.i]
            column_density = self.gible_rays.column_densities[self.i]
            n_intersections = self.gible_rays.n_all_clouds[self.i]
            flux_fields = self.flux_fields

        if velocity:
            x_field = self.velocity_field
            ax.set_xlim(self.velocity_field.min(), self.velocity_field.max())
            ax.set_xlabel('Velocity [km/s]')
            #if cloud_vels:
            #    for cloud in clouds:
            #        plt.plot(2*[cloud.vz], [1-cloud.m_scaled, 1], color='k', \
            #                 alpha=cloud.m_scaled/100)
        else:
            x_field = self.lambda_field
            ax.set_xlim(self.lambda_field.min(), self.lambda_field.max())
            ax.set_xlabel('$\lambda$ [Angstrom]')

        if component:
            # Method for using arbitrary colormap as color cycle

            if(self.gible_only):
                n = len(self.mask)
                color = plt.cm.viridis(np.linspace(0,0.7,n))
                ax.set_prop_cycle('color', color)

                for j, k in enumerate(self.mask):
                    p = ax.plot(x_field, flux_fields[j,:])
                    if annotate:
                        # Deal with cutoff annotations
                        if j==39:
                            text = "and %d more..." % (len(self.mask) - j)
                            xcoord = 0.02 + 0.70*(np.floor(j/20))
                            ycoord = 0.78 - 0.04*j + 0.80*np.floor(j/20)
                            ax.text(xcoord, ycoord, text, size='small', weight='bold',
                                color='k',transform=ax.transAxes)
                            continue
                        if j>39: continue
                        text = "%.1g M$_{\odot}$ %2d km/s" % (self.gible_clouds.masses.to('Msun')[k], self.gible_clouds.velocities.to('km/s')[k,2])
                        cindex = p[0].get_color()
                        xcoord = 0.02 + 0.70*(np.floor(j/20))
                        ycoord = 0.78 - 0.04*j + 0.80*np.floor(j/20)
                        ax.text(xcoord, ycoord, text, size='medium', weight='bold',
                                color=cindex,transform=ax.transAxes)

            else:
                n = int(n_intersections)
                color = plt.cm.viridis(np.linspace(0,0.7,n))
                ax.set_prop_cycle('color', color)

                for j in range(int(n_intersections)):
                    p = ax.plot(x_field, flux_fields[j,:])
                    if annotate:
                        # Deal with cutoff annotations
                        if j==39:
                            text = "and %d more..." % (n_intersections - j)
                            xcoord = 0.02 + 0.70*(np.floor(j/20))
                            ycoord = 0.78 - 0.04*j + 0.80*np.floor(j/20)
                            ax.text(xcoord, ycoord, text, size='small', weight='bold',
                                color='k',transform=ax.transAxes)
                            continue
                        if j>39: continue
                        #text = "%.1g M$_{\odot}$ %2d km/s" % (self.gible_clouds.masses.to('Msun')[k], self.gible_clouds.velocities.to('km/s')[k,2])
                        text = "%.1g $cm^{-2}$ %2d km/s" % (self.gible_rays.indiv_N[self.i,j], self.gible_rays.indiv_dv[self.i,j])
                        cindex = p[0].get_color()
                        xcoord = 0.02 + 0.70*(np.floor(j/20))
                        ycoord = 0.78 - 0.04*j + 0.80*np.floor(j/20)
                        ax.text(xcoord, ycoord, text, size='medium', weight='bold',
                                color=cindex,transform=ax.transAxes)


        # Finally plot the full spectrum
        # Sample every 2 elements to degrade resolution to HIRES
        #ax.plot(x_field, flux_field, 'k') # full resolution HIRES/2
        ax.step(x_field[::2], flux_field[::2], 'k', where='mid', linewidth=2) #HIRES resolution

        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Relative Flux')
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        if velocity == True:
            ax.set_xlim(-150, 150)
        if annotate:
            if mean:
                text1 = 'Mean: Num Clouds = %2d' % n_intersections
            else:
                #text1 = '#%04d: Num Clouds = %2d' % (self.i, n_intersections)
                text1 = 'Number of Cloudlets = %2d' % n_intersections
            ax.text(0.02, 0.95, text1, size='medium', weight='bold', color='black',transform=ax.transAxes)
            text2 = 'Column Density = %3.1g ${\\rm cm^{-2}}$' % column_density
            ax.text(0.02, 0.90, text2, size='medium', weight='bold', color='black',transform=ax.transAxes)
            text3 = 'EW = %4.2f ${\\rm \AA}$' % EW_val
            ax.text(0.02, 0.85, text3, size='medium', weight='bold', color='black',transform=ax.transAxes)
            #if self.subcloud_turb:
            #    text3 = 'Subcloud Turbulence'
            #    ax.text(0.98, 0.95, text3, size='medium', weight='bold', color='black', horizontalalignment='right', transform=ax.transAxes)

        plt.tight_layout()
        if filename is None:
            if mean:
                filename = 'gible_spectrum_mean.png'
            else:
                filename = 'gible_spectrum_%g_%04d.png' % (self.instr_resolution, self.i)

        plt.savefig(filename)
        plt.close()

    def generate_ray_sample(self, N, center, radius, first_n_to_plot=0,
                                component=True, ray_plot=True, attach_spectra=False):
            """
            Create a sample of N rays spanning an aperture with center and radius specified.
            Calculate their spectra passing through the associated cloud distribution.
            Returns set of rays (with EWs, clouds intersected, and column densities).
            """
            rays = GIBLERays(N, center, radius, generate=True)

            # Determine Mg II ion abundance based on cool gas number density
            # mgII_frac = 0.33 # Accurate for n_cl = 0.01
            ### NEED TO ADJUST THIS
            mgII_frac = 0.33 ### calc_mgII_frac(self.clouds.params['n_cl'])
            print("mgII frac = %f" % mgII_frac)

            for i in tqdm(range(N), "Generating Rays"):
                self.make_spectrum(rays, i, mgII_frac, attach=attach_spectra)
                # Plot spectrum after applying line spread function
                if i < first_n_to_plot and self.EW > 0:
                    self.plot_spectrum(component=True)
                self.clear_spectrum()
            self.clear_spectrum(total=True)
            return rays


def plot_rperp_ew(rays_perp, gible_rays, outfil='gible_GIBLErays_rperp_ew2796.pdf'):

    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(6,5))
    ax = plt.subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
    ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')

    lim = (gible_rays.EWs < 0.01)
    uplimEW = np.zeros(len(gible_rays.EWs[lim])) + 0.02
    ax.errorbar(rays_rperp[lim], uplimEW+np.random.normal(0,0.001,len(uplimEW)), \
        yerr=0.002, uplims=True, color='k', linestyle='none', alpha=0.3)  

    print("Number of detections = ", len(gible_rays.EWs[(gible_rays.EWs > 0.01)]))

    ax.plot(rays_rperp, gible_rays.EWs, linestyle='none', marker='o', color='k', ms=3, alpha=0.5)
    ax.set_xlabel(r'$R_{\rm perp}$')
    ax.set_ylabel(r'$W^{\rm CGM}_{2796} \rm (\AA)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(2,300)
    ax.set_ylim(0.01,6)

    plt.tight_layout()
    plt.savefig(outfil, format='pdf')

    ## MCLMIN0.1 version has 185 detections
    ## MCLMIN10 version has 164 detections
    ## MCLMIN1000 version has 126 detections


if __name__ == '__main__':

    # Load GIBLE clouds
    #filename = '/Users/krubin/Research/GPG/GIBLE/Inputs/CloudCatalog_S167RF4096_z0.hdf5'
    filename = '/Users/krubin/Research/GPG/GIBLE/Inputs/CloudCatalog_S98RF512_z0.hdf5'

    with h5py.File(filename, 'r') as f:
        gb_cloudPos = f["cloudPos"][:]            # cloud center of mass in kpc; z=0 plane aligned with galactic disk
        gb_cloudVel = f["cloudVel"][:]            # cloud 3D velocity in km/s
        gb_cloudVelDisp = f["cloudVelDisp"][:]    # cloud velocity dispersion
        gb_cloudMass = f["cloudMass"][:]          # total mass of cloud, solar masses
        gb_cloudNumCells = f["cloudNumCells"][:]  # number of cells in cloud
        gb_cloudMetal = f["cloudMetal"][:]           # EMPTY?? - mean metallicity of cloud in units of Zsun
        gb_cloudSize = f["cloudSize"][:]          # size computed as (3*volume / 4pi)^1/3 in kpc


    gible_clouds = GIBLEClouds(centers=gb_cloudPos, velocities=gb_cloudVel, \
        velocity_dispersions=gb_cloudVelDisp, masses=gb_cloudMass, \
        number_of_cells=gb_cloudNumCells, metallicities=gb_cloudMetal, radii=gb_cloudSize)
    gible_clouds.plot_clouds()
    gible_clouds.plot_parameters_space()

    # Describe available complex suite -- these go with MTOT_MCLMIN0.1 and MTOT_MCLMIN10
    #lgmtot_low = 2
    #lgmtot_high = 8
    #mtots = np.logspace(lgmtot_low, lgmtot_high, num=16)
    #mtots = mtots[:-2]

    # These go with MTOT_DCLMAX_MCLMIN10 and 1000
    mtots = np.array([1.0e2, 3.0e2, 1.0e3, 3.0e3, \
                            1.0e4, 3.0e4, 1.0e5, 3.0e5, \
                            1.0e6, 3.0e6, 1.0e7, 3.0e7])

    dclmaxs = [[0.05, 0.1, 0.5], [0.05, 0.1, 0.5], [0.08, 0.2, 0.4], [0.1, 0.2, 0.5], \
                    [0.15, 0.3, 0.8], [0.2, 0.4, 0.8], [0.3, 0.7, 1.0], [0.6, 1.0, 2.0], \
                    [1.0, 2.0], [2.0, 3.0], [5.0], [7.0]]

    # These go with MTOT_DCLMAX_MCLMIN0.1
    #mtots = np.array([1.0e2, 3.0e2, 1.0e3, 3.0e3, \
    #                        1.0e4, 3.0e4, 1.0e5, 3.0e5, \
    #                        1.0e6, 3.0e6, 1.0e7])

    #dclmaxs = [[0.05, 0.1, 0.5], [0.05, 0.1, 0.5], [0.08, 0.2, 0.4], [0.1, 0.2, 0.5], \
    #                [0.15, 0.3, 0.8], [0.2, 0.4, 0.8], [0.3, 0.7, 1.0], [0.6, 1.0, 2.0], \
    #                [1.0, 2.0], [2.0, 3.0], [5.0]]

    np.random.seed(seed=int(12))
    Nsight = 20
    sg = GIBLESpectrumGenerator(gible_clouds, debug=True, \
        complex_path='/Users/krubin/Research/GPG/GIBLE/GIBLE_Complex_Suite/MTOT_DCLMAX_MCLMIN10/', \
        complex_mtots=mtots, complex_dclmaxs=dclmaxs, gible_only=False) #instr_pixel_width=0.5, instr_resolution=190000.0)
    gible_rays = sg.generate_ray_sample(Nsight, [0,0], 200, first_n_to_plot=10, attach_spectra=True)
    gible_rays.save('GIBLErays_testing_spec_save.h5')

    gible_rays = GIBLERays(generate=False)
    #gible_rays.print_ascii('GIBLErays_testing_spec_save.h5', updir='RaysAscii')
    #gible_rays.save('GIBLErays_MTOT_DCLMAX_MCLMIN10_190000.h5')
    
    
    gible_rays = GIBLERays(generate=False)
    gible_rays.load('GIBLErays_MTOT_DCLMAX_MCLMIN10.h5')

    rays_rperp = np.sqrt(gible_rays.coords[:,0]**2 + gible_rays.coords[:,1]**2)
    plot_rperp_ew(rays_rperp, gible_rays, outfil='gible_GIBLErays_MTOT_DCLMAX_MCLMIN10_rperp_ew2796.pdf')

    



    ###########################################################
    # Fig for talk
    colors = plt.cm.viridis(np.linspace(0,0.90,5))       

    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(1,5,width_ratios=[1,1,1,0.2,1],wspace=0.0)

    ax = fig.add_subplot(gs[0,0])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
    ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')

    ax.set_xlabel(r'$R_{\bot}$ (kpc)')
    ax.set_ylabel(r'$W^{\rm CGM}_{2796} \rm (\AA)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(4,300)
    ax.set_ylim(0.01,4)

    gible_rays = GIBLERays(generate=False)
    gible_rays.load('GIBLErays_MTOT_DCLMAX_MCLMIN0.1.h5')
    rays_rperp = np.sqrt(gible_rays.coords[:,0]**2 + gible_rays.coords[:,1]**2)

    lim = (gible_rays.EWs < 0.01)
    uplimEW = np.zeros(len(gible_rays.EWs[lim])) + 0.02
    ax.errorbar(rays_rperp[lim], uplimEW+np.random.normal(0,0.001,len(uplimEW)), \
        yerr=0.002, uplims=True, color=colors[1], linestyle='none', alpha=0.3)  
    print("Number of detections = ", len(gible_rays.EWs[(gible_rays.EWs > 0.01)]))
    ax.plot(rays_rperp, gible_rays.EWs, linestyle='none', marker='o', color=colors[1], ms=3, alpha=0.5)
    ax.text(0.57,0.93, r'$m_{\rm cl,min} = 10^{-1} M_{\odot}$', transform=ax.transAxes, fontsize=10)

    dumxgt10 = np.arange(10,300)
    logW_dutta = -0.20 + (-0.009*dumxgt10)
    ax.plot(dumxgt10, 10.0**logW_dutta, ls='dotted', color='lightblue', lw=2, label='Dutta et al. 2020')

    logW_dutta_up = -0.20+0.42 + ((-0.009+0.003)*dumxgt10)
    logW_dutta_lo = -0.20-0.39 + ((-0.009-0.003)*dumxgt10)
    ax.fill_between(dumxgt10, 10.0**logW_dutta_lo, 10.0**logW_dutta_up, color='lightblue', alpha=0.2)


    ax = fig.add_subplot(gs[0,1])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
    ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')

    ax.set_xlabel(r'$R_{\bot}$ (kpc)')
    #ax.set_ylabel(r'$W^{\rm CGM}_{2796} \rm (\AA)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(4,300)
    ax.set_ylim(0.01,4)
    ax.yaxis.set_tick_params(labelleft=False)

    gible_rays = GIBLERays(generate=False)
    gible_rays.load('GIBLErays_MTOT_DCLMAX_MCLMIN10.h5')
    rays_rperp = np.sqrt(gible_rays.coords[:,0]**2 + gible_rays.coords[:,1]**2)

    lim = (gible_rays.EWs < 0.01)
    uplimEW = np.zeros(len(gible_rays.EWs[lim])) + 0.02
    ax.errorbar(rays_rperp[lim], uplimEW+np.random.normal(0,0.001,len(uplimEW)), \
        yerr=0.002, uplims=True, color=colors[2], linestyle='none', alpha=0.3)  
    print("Number of detections = ", len(gible_rays.EWs[(gible_rays.EWs > 0.01)]))
    ax.plot(rays_rperp, gible_rays.EWs, linestyle='none', marker='o', color=colors[2], ms=3, alpha=0.5)
    ax.text(0.57,0.93, r'$m_{\rm cl,min} = 10 M_{\odot}$', transform=ax.transAxes, fontsize=10)

    dumxgt10 = np.arange(10,300)
    logW_dutta = -0.20 + (-0.009*dumxgt10)
    ax.plot(dumxgt10, 10.0**logW_dutta, ls='dotted', color='lightblue', lw=2, label='Dutta et al. 2020')

    logW_dutta_up = -0.20+0.42 + ((-0.009+0.003)*dumxgt10)
    logW_dutta_lo = -0.20-0.39 + ((-0.009-0.003)*dumxgt10)
    ax.fill_between(dumxgt10, 10.0**logW_dutta_lo, 10.0**logW_dutta_up, color='lightblue', alpha=0.2)


    ax = fig.add_subplot(gs[0,2])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(which='major', axis='both', width=1.5, length=5, top='on', right='on', direction='in')
    ax.tick_params(which='minor', axis='both', width=1.5, length=3, top='on', right='on', direction='in')

    ax.set_xlabel(r'$R_{\bot}$ (kpc)')
    #ax.set_ylabel(r'$W^{\rm CGM}_{2796} \rm (\AA)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(4,300)
    ax.set_ylim(0.01,4)
    ax.yaxis.set_tick_params(labelleft=False)

    gible_rays = GIBLERays(generate=False)
    gible_rays.load('GIBLErays_MTOT_DCLMAX_MCLMIN1000.h5')
    rays_rperp = np.sqrt(gible_rays.coords[:,0]**2 + gible_rays.coords[:,1]**2)

    lim = (gible_rays.EWs < 0.01)
    uplimEW = np.zeros(len(gible_rays.EWs[lim])) + 0.02
    ax.errorbar(rays_rperp[lim], uplimEW+np.random.normal(0,0.001,len(uplimEW)), \
        yerr=0.002, uplims=True, color=colors[3], linestyle='none', alpha=0.3)  
    print("Number of detections = ", len(gible_rays.EWs[(gible_rays.EWs > 0.01)]))
    ax.plot(rays_rperp, gible_rays.EWs, linestyle='none', marker='o', color=colors[3], ms=3, alpha=0.5)

    dumxgt10 = np.arange(10,300)
    logW_dutta = -0.20 + (-0.009*dumxgt10)
    ax.plot(dumxgt10, 10.0**logW_dutta, ls='dotted', color='lightblue', lw=2, label='Dutta et al. 2020')
    ax.text(0.57,0.93, r'$m_{\rm cl,min} = 10^{3} M_{\odot}$', transform=ax.transAxes, fontsize=10)

    logW_dutta_up = -0.20+0.42 + ((-0.009+0.003)*dumxgt10)
    logW_dutta_lo = -0.20-0.39 + ((-0.009-0.003)*dumxgt10)
    ax.fill_between(dumxgt10, 10.0**logW_dutta_lo, 10.0**logW_dutta_up, color='lightblue', alpha=0.2)


    plt.tight_layout()
    plt.savefig('gible_GIBLErays_MTOT_DCLMAX_MCLMINall_rperp_ew2796.pdf', format='pdf')
   
    embed()


    ###########################################################################
    ################# Scratch #################################################
    ###########################################################################


    # Place sightline within Rperp < 200 kpc
    # coords = generate_random_coordinates(Nsight, center, Rperp)

    #for i in range(Nsight):

        # Calculate intersection with GIBLE clouds - from cloudflex.calculate_intersections
    #    d = np.linalg.norm(coords[i] - gb_cloudPos[:, :2], axis=1)

        # all clouds that are intersected have d<r
        # when d>r, this raises " RuntimeWarning: invalid value encountered in sqrt"
        # so suppressing that temporarily
    #    with warnings.catch_warnings():
    #        warnings.simplefilter("ignore")
    #        pathlengths = 2*(gb_cloudSize**2 - d**2)**0.5

        # when d>r, pathlengths will be imaginary
    #    dls = np.nan_to_num(pathlengths) * kpc
    #    mask = np.nonzero(dls)[0]

    #    column_densities = dls.to('cm') * gb_cloudNumDens
    #    mg_column_densities = solar_abundance['Mg'] * column_densities * Z_cl * X_hydrogen
    #    mgII_column_densities = mg_column_densities * mgII_frac

    #    tau_field = np.zeros_like(lambda_field)
    #    tau_fields = np.zeros([len(mask), lambda_field.shape[0]])

    #    for index, j in enumerate(mask):
    #        tau_fields[index,:] += deposit_voigt(dls[j], gb_cloudVel[j,2], \
    #            mgII_column_densities[j], \
    #            lambda_0, f_value, gamma, \
    #            gb_cloudVelDisp[j], lambda_field)

    
    