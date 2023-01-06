"""
Script to take as input N-body fields, and create dataset of pairs of 
highly-correlated lognormal and N-body slices. Code is not extremely factorised, 
and requires some pre-computed density fields to be run smoothly.
Get in touch with dr.davide.piras@gmail.com if you need help with this script.
"""

import numpy as np
import gc
from nbodykit.lab import ArrayMesh, FFTPower, FFTCorr, cosmology, LinearMesh
from nbodykit.cosmology.correlation import pk_to_xi, xi_to_pk
from scipy.interpolate import interp1d
from classy import Class
import tensorflow as tf

def calc_ps_from_field_nbodykit(field, BoxSize=[1000, 1000], kmin=kmin, kmax=kmaxes[0], dk=7e-3):
    field_mesh = ArrayMesh(field, BoxSize=BoxSize)
    r_2d = FFTPower(field_mesh, mode='1d', kmin=kmin, kmax=kmax)#, dk=dk)
    return r_2d.power

def remove_nans(x):
    return x[~np.isnan(x)]

def shift(field, x=0, y=0, random=False, seed=42):
    # random controls whether the shift should be random or not
    # if True, x and y will be ignored 
    if random:
        np.random.seed(seed)
        side = field.shape[0]
        x, y = np.random.choice(np.arange(-side, side), 2)
    field_shifted = np.roll(field, (x, y), axis=(0, 1))
    return field_shifted, (x, y)

def shift_all_slices_2D(fieldA, fieldB):
    # we need to shift both ICs and final slices by the same amount
    shift_fieldA = np.empty_like(fieldA)
    shift_fieldB = np.empty_like(fieldB)
    # we shift along x as it's easiest
    current_seed = np.random.randint(1e9)
    shift_fieldA[:, :], _ = shift(fieldA[:, :], random=True, seed=current_seed)
    shift_fieldB[:, :], _ = shift(fieldB[:, :], random=True, seed=current_seed) 
    return shift_fieldA, shift_fieldB

k0 = 1e-5 # in h/Mpc
kmin = 0.025 # in h/Mpc
# we try multiple kmax values, to avoid numerical errors
kmaxes = np.concatenate((np.arange(0.30, 0.35, 0.005), np.arange(0.295, 0.25, -0.005))) # in h/Mpc
n_files = 1000 # will have to be e.g. 800 or 1000; the total number of simulations considered
field_side = 128 # the side of each slice; either 128 or 512
counter = 0 # to count how many NaNs were found in the end
fudge = True
max_fudges = 100  # some nans will happen because fewer fudges are needed; we'll remove these nans when training, they are caused by numerical errors due to high noise at low k end
saving_step = 1 # to control size of output files

N = n_files*field_side # total number of images
total_input_dataset = np.empty((N, field_side, field_side, 1))
total_output_dataset = np.empty((N, field_side, field_side, 1))

lh_cosmos = np.loadtxt('latin_hypercube_params.txt') # used for the latin-hypercube case; just use fiducial cosmology alternatively

for i in range(n_files):
    # load density fields; THESE ARE NOT PROVIDED IN THIS REPOSITORY
    df_path_z0 = f'data_path/{i}/df_m_{field_side}_z=0.npy'
    df_path_z1 = f'data_path/{i}/df_m_{field_side}_z=127.npy'
    output_train = np.load(df_path_z0)
    input_train = np.load(df_path_z1)

    # calculate theory ps; used for better numerical stability
    cosmo_class = Class()
    hubble = lh_cosmos[i, 2] # 0.6711
    omega_b = lh_cosmos[i, 1] # 0.049
    omega_cdm = lh_cosmos[i, 0]-omega_b
    ns = lh_cosmos[i, 3] # 0.9624
    sigma8 = lh_cosmos[i, 4] # 0.834
    parameters = [omega_b*hubble**2, omega_cdm*hubble**2, hubble, ns, sigma8]
    cosmo_params = {'omega_b': parameters[0],
                 'omega_cdm': parameters[1],
                 'h': parameters[2],
                 'n_s': parameters[3],
                 'sigma8': parameters[4],
                 'output': 'mPk',
                 #'c_min': parameters[5],
                 'non linear': 'hmcode',
                 'z_max_pk': 50,
                 'P_k_max_h/Mpc': 30,
                 #'Omega_k': 0.,
                 #'N_ncdm' : 0,
                 #'N_eff' : 3.046,
                }

    cosmo_class.set(cosmo_params)
    try:
        cosmo_class.compute()
        # fails sometimes since omega_b is too high...
    except:
        # if it fails, just use fiducial cosmology; results do not change much
        print(i)
        omega_b = 0.049
        omega_cdm = lh_cosmos[i, 0]-omega_b
        parameters = [omega_b*hubble**2, omega_cdm*hubble**2, hubble, ns, sigma8]
        cosmo_params = {'omega_b': parameters[0],
                  'omega_cdm': parameters[1],
                  'h': parameters[2],
                  'n_s': parameters[3],
                  'sigma8': parameters[4],
                  'output': 'mPk',
                  #'c_min': parameters[5],
                  'non linear': 'hmcode',
                  'z_max_pk': 50,
                  'P_k_max_h/Mpc': 30,
                  #'Omega_k': 0.,
                  #'N_ncdm' : 0,
                  #'N_eff' : 3.046,
                 }
        cosmo_class.set(cosmo_params)
        cosmo_class.compute()

    for j in range(field_side):
        # first, calculate ps of current slice
        for kmax in kmaxes:
            P2D_128 = calc_ps_from_field_nbodykit(output_train[j, :, :], kmin=kmin, kmax=kmax)
            
            k_values = remove_nans(P2D_128['k'])
            ps_values = remove_nans(P2D_128['power'].real)

            # we also add some values from k0 to kmin, from the theory
            k_values_theory = np.logspace(np.log10(k0*hubble), np.log10(k_values[0]*hubble)-1e-10, 500) # these are in 1/Mpc
            power_spectrum_theory = np.empty(500)
            for ind in range(500):
                power_spectrum_theory[ind] = cosmo_class.pk(k_values_theory[ind], 0)
            k_values_theory /= hubble  # so that it is in h/Mpc
            power_spectrum_theory *= hubble**3  # so that it is in Mpc/h**3

            k_values_interp = np.concatenate([k_values_theory, k_values])
            ps_values_interp = np.concatenate([power_spectrum_theory*ps_values[0]/power_spectrum_theory[-1], ps_values])

            n_points = 5000 # k_values.shape[0] seems to fail!
            f2 = interp1d(k_values_interp, ps_values_interp, kind='linear')
            k_values_ = np.logspace(np.log10(k_values_interp[0]+1e-10), np.log10(k_values[-1]-1e-10), n_points)
            power_spectrum = np.empty(n_points)
            for ind in range(n_points):
                power_spectrum[ind] = f2(k_values_[ind])

            cf_class = pk_to_xi(k_values_, power_spectrum, extrap=True)
            r = np.logspace(-5, 5, int(1e5))

            def cf_g(r):
                return np.log(1+cf_class(r))

            # then it should be easy to use the same transformation as above, just inverse, to obtain a Gaussian-like power spectrum
            Pclass_g = xi_to_pk(r, cf_g(r))

            # using the input slice at z=127 to create the input dataset to U-net
            # we combine the phase information there with the power from the "theory"
            g_field = LinearMesh(Pclass_g, Nmesh=[field_side, field_side], BoxSize=1000, seed=np.random.randint(1e9), unitary_amplitude=True).preview() - 1 
            ft_g = np.fft.fftn(g_field)
            abses_g = np.abs(ft_g)

            ft_IC = np.fft.fftn(input_train[j, :, :])
            abses_ic = np.abs(ft_IC)
            ft_mixed = ft_IC / abses_ic * abses_g
            # note that this inverse transform yields some 1e-17 imaginary parts, which I think are just
            # numerical errors; np.real_if_close takes care of that.
            g_field_mixed = np.real_if_close(np.fft.ifftn(ft_mixed), tol=1000)
            # finally, take the LN off this Gaussian map
            gaussian_stddev = np.std(g_field_mixed.flatten())
            ln_field_mixed = np.exp(g_field_mixed-gaussian_stddev**2/2)-1
            # now we have the input and output boxes; we need to shift them, and then normalise each slice
            # before that, we calculate the power spectrum of each input and output box
            # since we usually need to add the fudge here
            
            in_ps = calc_ps_from_field_nbodykit(ln_field_mixed, kmin=k0, kmax=kmax)
            out_ps = calc_ps_from_field_nbodykit(output_train[j, :, :], kmin=k0, kmax=kmax) 
            k_values_in = in_ps['k']
            k_values_out = out_ps['k']

            if np.isnan(np.sum(in_ps['power'].real)):
                # breaking even before fudging, for nans due to kmax
                continue

            if not fudge:
                # just conclude
                in_ps_all.append(in_ps['power'].real)
                out_ps_all.append(out_ps['power'].real)

            if fudge:
                power_spectrum_adjusted = power_spectrum
                fudge_done = False # we fudge until we are within the threshold level
                threshold = 0.001 # 0.1% in this case, quite conservative
                fudge_counter = 0
                while fudge_done == False:

                    test_adjust = out_ps['power'].real/in_ps['power'].real 
                    
                    k_val_interp = interp1d(k_values_in, test_adjust, kind='linear')
                    k_pivot = k_values_[np.argwhere(k_values_ >= k_values_in[0])[:-1]]
                    interpolated_k_part =  k_val_interp(k_pivot)[:, 0]
                    interpolated_k_part = np.concatenate((interpolated_k_part, interpolated_k_part[-1:]))

                    pivot = int(np.argwhere(k_values_ >= k_values_in[0])[0])

                    right_hand_side_power_spectrum = np.multiply(power_spectrum_adjusted[pivot:], interpolated_k_part)
                    # for some reason, while correct this seems to give some nans every now and then, so be careful about it
                    power_spectrum_adjusted = np.concatenate((power_spectrum_adjusted[:pivot]*right_hand_side_power_spectrum[np.argwhere(k_pivot <= kmin)].mean()/power_spectrum_adjusted[:pivot][-1], right_hand_side_power_spectrum))

                    # same as above, using the adjusted ps
                    cf_class = pk_to_xi(k_values_, power_spectrum_adjusted, extrap=True)
                    r = np.logspace(-5, 5, int(1e5))

                    def cf_g(r):
                        return np.log(1+cf_class(r))

                    # then it should be easy to use the same transformation as above, just inverse, to obtain a Gaussian-like power spectrum
                    Pclass_g = xi_to_pk(r, cf_g(r))

                    # using the input slice at z=127 to create the input dataset to U-net
                    # we combine the phase information there with the power from the "theory"
                    #np.random.randint(1e9)
                    g_field = LinearMesh(Pclass_g, Nmesh=[field_side, field_side], BoxSize=1000, seed=np.random.randint(1e9), unitary_amplitude=True).preview() - 1 
                    
                    # can add a check to stop if too many fudges yield a nan
                    #if np.isnan(g_field.sum()):
                    #    print(i, j, fudge_iter)
                    #    break
                    
                    ft_g = np.fft.fftn(g_field)
                    abses_g = np.abs(ft_g)
                    ft_mixed = ft_IC / abses_ic * abses_g
                    # note that this inverse transform yields some 1e-17 imaginary parts, which I think are just
                    # numerical errors; np.real_if_close takes care of that.
                    g_field_mixed = np.real_if_close(np.fft.ifftn(ft_mixed), tol=1000)
                    # finally, take the LN off this Gaussian map
                    gaussian_stddev = np.std(g_field_mixed.flatten())
                    ln_field_mixed = np.exp(g_field_mixed-gaussian_stddev**2/2)-1

                    in_ps = calc_ps_from_field_nbodykit(ln_field_mixed, kmin=k0, kmax=kmax)
                    out_ps = calc_ps_from_field_nbodykit(output_train[j, :, :], kmin=k0, kmax=kmax) 
                    k_values_in = in_ps['k']
                    k_values_out = out_ps['k']
                   
                    # we check if the maximum discrepancy is less than 0.1%, otherwise keep on fudging
                    # in some cases there will be discrepancies at low k which are hard to reduce, so
                    # we set a maximum of 100 fudges. We only look at k > 0.2, since we never use the information below that.
                    max_discrepancy = np.max(np.abs(out_ps['power'].real[3:] - in_ps['power'].real[3:]) / in_ps['power'].real[3:])
                    if np.isnan(max_discrepancy):
                        print(max_discrepancy, j)
                        break
                    if max_discrepancy <= threshold:
                        print(max_discrepancy, j)
                        fudge_done = True
                    else:
                        fudge_counter += 1
                    if fudge_counter >= max_fudges:
                        print(max_discrepancy, j)
                        break 


            if not np.isnan(np.sum(in_ps['power'].real)):
                break
        

        if np.isnan(np.sum(in_ps['power'].real)):
            counter += 1
            print(i, j)

        # shift slices along one direction to remove correlations between slices
        shifted_input_train, shifted_output_train = shift_all_slices_2D(ln_field_mixed, output_train[j, :, :])
            
        total_input_dataset[i*field_side+j, :, :, 0] = np.copy(shifted_input_train)
        total_output_dataset[i*field_side+j, :, :, 0] = np.copy(shifted_output_train)

    gc.collect() # memory management

    # save, using TF datasets
    # since saving_step is 1, we essentially save 1 file per simulation
    if (i+1)%saving_step == 0:
        print(f'Saving file {i+1}')
        tf.data.experimental.save(tf.data.Dataset.from_tensor_slices((total_input_dataset[(i-saving_step+1)*field_side:(i+1)*field_side], total_output_dataset[(i-saving_step+1)*field_side:(i+1)*field_side])), f'data_path/{i}/tf_dataset', compression='GZIP')
        gc.collect()



