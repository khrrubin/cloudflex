import h5py
import numpy as np
from unyt import cm, K, proton_mass
import os
import glob
from IPython import embed

from cloudflex import \
    SpectrumGenerator, \
    plot_histogram, \
    plot_clouds, \
    Clouds, \
    Rays, \
    create_and_plot_clouds, \
    plot_covering_frac, \
    plot_clouds_buffer, \
    make_quad_plot
    
from observe_halo import GIBLEClouds


##############################################################
######## Complexes get generated here  #######################
##############################################################    

def make_complex_for_gible_cloud(flg='MCLMIN', uppath='./GIBLE_Complex_Suite', haloname='blank', \
    cloud_indx=0, mass=1e6, number_density=1e-2 / cm**3, radius=5.0, metallicity=0.33):

    if not os.path.exists(uppath):
        os.makedirs(uppath)
    os.chdir(uppath)
    
    if not os.path.exists('./'+haloname):
        os.makedirs(haloname)
    os.chdir(haloname)

    # Store everything in parameters dictionary to pass to Clouds class
    params = {}

    # Turbulence Generation Stuff
    N = 128                             # Default: 128
    seed = int(2) # random seed
    np.random.seed(seed=seed)
    dtype = np.float64 # data precision

    params['seed'] = seed
    params['N'] = N
    params['n'] = [N, N, N]
    params['kmin'] = int(1)             # Default: 1
    params['kmax'] = int(N/2)           # Default: N/2
    params['f_solenoidal'] = 2/3.       # Default: 2/3
    params['beta'] = 1/3.               # Default: 1/3
    params['vmax'] = 30                 # Default: 30 km/s

    #### Cloud Generation Stuff
    params['alpha'] = 2                 # Default: 2
    params['zeta'] = 1                 # Default: 1

    # cloud mass min/max (Msun)
    params['mclmin'] = 1e1              # Default: 10 Msun
    params['mclmax'] = mass             # Default: 100000 Msun 

    # cloud position from center min/max (kpc)
    params['dclmin'] = 0.001            # Default: 0.1 kpc
    params['dclmax'] = radius           # Default: 5 kpc
    params['center'] = [0.0, 0.0]       # Default: (0, 0)

    # background density and temperature of hot medium
    # density and temperature of cool clouds
    params['T_cl'] = 1e4 * K            # Default: 1e4 K
    #params['n_cl'] = 1e-2 / cm**3      # Default: 1e-2 cm^-3
    params['n_cl'] = number_density
    params['Z_cl'] = metallicity        # Default: 0.33
    params['rho_cl'] = params['n_cl'] * proton_mass
    params['total_mass'] = mass         # Default: 1e6 Msun
    params['clobber'] = True            # Default: True

    params['gible_index'] = cloud_indx
    params['gible_halo_name'] = haloname

    # Ensuring seed state is same for each iteration
    state = np.random.get_state()

    if(flg=='MCLMIN'):

        if not os.path.exists(flg):
            os.makedirs(flg)
        os.chdir(flg)

        ## Mcl,min loop
        lgmclmin_low = -2
        lgmclmin_high = 5
        mclmins = np.logspace(lgmclmin_low, lgmclmin_high, num=8)
    
        for i in range(len(mclmins)):
            np.random.set_state(state)
            mclmin = mclmins[i]

            if (mclmin > params['mclmax']):
                continue

            else:

                print("Generating clouds for mclmin %.1g" % mclmin)
                if mclmin < 1:
                    path = './%.1g' % mclmin
                    cloud_fil = 'clouds_gb_%d_mclmin_%.1g.h5' % (cloud_indx, mclmin)
                else:
                    path = './%1d' % mclmin
                    cloud_fil = 'clouds_gb_%d_mclmin_%1d.h5' % (cloud_indx, mclmin)
                if not os.path.exists(path):
                    os.makedirs(path)
                os.chdir(path)
                params['mclmin'] = mclmin

                clouds = Clouds(params=params)
                clouds.generate_clouds()
                clouds.save(cloud_fil)
                
        
                os.chdir('..')

        os.chdir('..')
    os.chdir('../..')
    
    


def make_rays(uppath='./GIBLE_Complex_Suite', haloname='S98', par_flg='MCLMIN', overwrite=False):

    os.chdir(os.path.join(uppath,haloname,par_flg))
    dirs = os.listdir(path='.')

    for dir in dirs:
        os.chdir(dir)
        clouds_files = glob.glob('clouds*.h5')

        for fil in clouds_files:
            namestr = fil[6:-3]
            rays_fil = 'rays%s.h5' % namestr

            if not (os.path.isfile(rays_fil)) or overwrite:
                create_rays(fil,rays_fil)
            else:
                print('Skipping ', rays_fil)

        os.chdir('..')

    os.chdir('../../..')


def create_rays(clouds_fil, rays_fil):
    ## Taken from make_rays.py

    clouds = Clouds()
    clouds.load(clouds_fil)
    params = clouds.params
    np.random.seed(seed=params['seed'])

    ## Generate and Plot Spectra for N random rays passing through domain
    ## Also plot distribution of clouds and rays

    # Create N random Rays passing through domain
    N = 10000
    sg = SpectrumGenerator(clouds, subcloud_turb=True)
    rays = sg.generate_ray_sample(N, params['center'],  params['dclmax'],
                                  first_n_to_plot=0, attach_spectra=False)
    rays.save(rays_fil)

    # Create histograms of EW, clouds intersected, and column density of all rays
    # plot_histogram(rays.column_densities, "column densities", clouds, log=True, xrange=[11,18])
    # plot_histogram(rays.EWs, "equivalent widths", clouds, xrange=[0, 1.0], n_bins=40)
            


###################################################################
######## Read in cloud catalog and set up complex suite ###########
###################################################################       

def setup_and_run_gible_suite(filename = '/Users/krubin/Research/GPG/GIBLE/Inputs/CloudCatalog_S98RF512_z0.hdf5', \
    haloname='S98', resolved_lim=10**5, par_flg='MCLMIN'):        

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

    for i, index in enumerate(gible_clouds.index):

        # Select resolved clouds
        if (gible_clouds.masses.to('Msun').value[i] > resolved_lim):

            print("Generating complexes for cloud index ", index, " of ", len(gible_clouds.index), " total clouds with mass = ", gible_clouds.masses.to('Msun')[i])
    
            make_complex_for_gible_cloud(flg=par_flg, uppath='./GIBLE_Complex_Suite', haloname=haloname, \
                cloud_indx=index, mass=gible_clouds.masses.to('Msun').value[i], \
                number_density=gible_clouds.number_densities.to(1.0/cm**3)[i], \
                radius=gible_clouds.radii.to('kpc').value[i], metallicity=gible_clouds.metallicities[i])


            
if __name__ == '__main__':

    # Load GIBLE clouds
    haloname = 'S98'
    resolved_lim = 10**5  # Need to decide on this!!
    filename = '/Users/krubin/Research/GPG/GIBLE/Inputs/CloudCatalog_S98RF512_z0.hdf5'

    setup_and_run_gible_suite(filename=filename, haloname=haloname, resolved_lim=resolved_lim, par_flg='MCLMIN')

    ## after running the above, and make the rays files:
    #make_rays(uppath='./GIBLE_Complex_Suite', haloname='S98', par_flg='MCLMIN', overwrite=False)

    embed()