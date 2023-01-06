import numpy as np
import Pk_library as PKL
# this is for the bispectra; contact Davide Piras to access this, since it is not public yet
import os, sys
sys.path.append('/home/dp1627/latin_hypercube/PiInTheSky/')
sys.path.append('/home/dp1627/latin_hypercube/PiInTheSky/BinnedPrimordialTemplates/')
import PiInTheSky.binnedEstimatorFullSky as binEstFullSky
import PiInTheSky.binnedEstimatorFlatSky as binEstFlatSky
import healpy as hp
import scipy.stats
from nbodykit.lab import *


def test_single_model(test_set, size=19):

    ps_values_train_in = []
    ps_values_train_out = []
    ps_values_train_predict = []

    kmin = 1e-5 # in h/Mpc
    kmax = 0.3 # in h/Mpc # apparently higher values fail

    def calc_ps_from_field_nbodykit(field, BoxSize=[1000, 1000], kmin=kmin, kmax=kmax, dk=1e-2):
        field_mesh = ArrayMesh(field, BoxSize=BoxSize)
        r_2d = FFTPower(field_mesh, mode='1d', kmin=kmin, kmax=kmax)#, dk=1e-4)
        return r_2d.power

    def remove_nans(x):
        return x[~np.isnan(x)]


    for i in range(3*size):

        P2D = calc_ps_from_field_nbodykit(test_set[i, ..., 0], kmin=kmin, kmax=kmax)

        k_values = remove_nans(P2D['k'])[3:]
        ps_values = remove_nans(P2D['power'].real)[3:]
        if i < size:
            ps_values_train_in.append(ps_values)
        elif i >= 2*size:
            ps_values_train_predict.append(ps_values)
        else:
            ps_values_train_out.append(ps_values)
        #print(i)

    diff = (np.mean(ps_values_train_predict, axis=0) - np.mean(ps_values_train_out, axis=0))/(np.mean(ps_values_train_out, axis=0))
    pred_std = np.std(ps_values_train_predict, axis=0)/np.sqrt(size)
    out_std = np.std(ps_values_train_out, axis=0)/np.sqrt(size)
    den = np.sqrt( (pred_std / np.mean(ps_values_train_out, axis=0))**2 + (np.mean(ps_values_train_predict, axis=0) * out_std / (np.mean(ps_values_train_out, axis=0)**2))**2 )

    a = np.mean(np.abs(diff)/den)

    histos = []
    for k in range(3*size):
        current_image = test_set[k, ..., 0]

        hist, bin_edges = np.histogram(current_image, bins=(np.logspace(0, 0.7)-2), range=(-1, 3))
        histos.append(hist)

    histos = np.array(histos)

    mean_bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2

    out_mean = np.mean(histos[size:2*size], axis=0)
    pred_mean = np.mean(histos[2*size:3*size], axis=0)
    diff = (pred_mean - out_mean)/(out_mean)
    pred_std = np.std(histos[2*size:3*size], axis=0)/np.sqrt(size)
    out_std = np.std(histos[size:2*size], axis=0)/np.sqrt(size)
    den = np.sqrt( (pred_std / out_mean)**2 + (pred_mean * out_std / (out_mean**2))**2 )

    b = np.mean(np.abs(diff[5:])/den[5:])
    
    def get_single_peak(matrix, i, j):
        region = matrix[i-1 : i+2,
                        j-1 : j+2]
        if matrix[i, j] >= region.all():
            return matrix[i, j]
        else:
            pass

    histos_peaks = []
    for k in range(3*size):
        current_image = test_set[k, ..., 0]
        # we pad so that we take into account the periodic boundary conditions
        current_image = np.pad(current_image, 1, mode='wrap')
        peaks = []
        for ii in range(128):
            for jj in range(128):
                # we start from (1,1) since we want to sart from the true original value, comparing to the neighbours
                single_peak = get_single_peak(current_image, 1+ii, 1+jj)
                if single_peak is not None:
                    peaks.append(single_peak)

        hist, bin_edges_peaks = np.histogram(peaks, bins=100, range=(-0.5, 3.5))
        histos_peaks.append(hist)

    histos_peaks = np.array(histos_peaks)

    mean_bin_edges_peaks = (bin_edges_peaks[1:] + bin_edges_peaks[:-1]) / 2

    out_mean = np.mean(histos_peaks[size:2*size], axis=0)
    pred_mean = np.mean(histos_peaks[2*size:3*size], axis=0)
    diff = (pred_mean - out_mean)/(out_mean)
    pred_std = np.std(histos_peaks[2*size:3*size], axis=0)/np.sqrt(size)
    out_std = np.std(histos_peaks[size:2*size], axis=0)/np.sqrt(size)
    den = np.sqrt( (pred_std / out_mean)**2 + (pred_mean * out_std / (out_mean**2))**2 )

    c = np.mean(np.abs(diff[40:])/den[40:])

    # in k units
    binEdges = np.arange(0.005,0.5,0.01)
    Nx = Ny = 128
    pixScaleX = pixScaleY = 1000 / Nx # 1 Gpc/h, divided by 128 pixels

    bispecs = []
    for i in range(3*size):
        field = test_set[i, ..., 0]
        field = np.fft.fft2(field)
        EST=binEstFlatSky.binnedEstimator(Nx,Ny,pixScaleX,pixScaleY,binEdges=binEdges,invC=0)
        bispec,_,_=EST.analyze([field], calcNorm=0)
        bispecs.append(bispec)
    bispecs = np.array(bispecs)

    k1 = 0.1 
    k2 = 0.3
    k3 = np.arange(k2-k1, k2+k1, 0.01)
    theta = np.arccos((k3**2 - k1**2 - k2**2)/(-2*k1*k2))
    theta[0] = 0
    theta[-1]= np.pi
    bin_mid_points = np.arange(0.01, 0.459, 0.01)

    def find_nearest_id(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    arg_k1 = find_nearest_id(bin_mid_points, k1)
    arg_k2 = find_nearest_id(bin_mid_points, k2)

    bispecs_in = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_in[j, ii] = bispecs[j+0, 0,0,0][indices[0], indices[1], indices[2]] 

    bispecs_out = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_out[j, ii] = bispecs[j+size, 0,0,0][indices[0], indices[1], indices[2]] 

    bispecs_pred = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_pred[j, ii] = bispecs[j+2*size, 0,0,0][indices[0], indices[1], indices[2]] 

        
    b_out_mean = np.mean(bispecs_out, axis=0)
    b_pred_mean = np.mean(bispecs_pred, axis=0)
    diff = (b_pred_mean - b_out_mean)/(b_out_mean)
    b_pred_std = np.std(bispecs_pred, axis=0)/np.sqrt(size)
    b_out_std = np.std(bispecs_out, axis=0)/np.sqrt(size)
    den = np.sqrt( (b_pred_std / b_out_mean)**2 + (b_pred_mean * b_out_std / (b_out_mean**2))**2 )
    d = np.mean(np.abs(diff)/den)

    # reduced bispectrum
    
    pss = []
    BoxSize = 1000.0 # size of the density field in Mpc/h
    MAS = 'PCS'

    for i in range(3*size):
        field = test_set[i, ..., 0]
        Pk2D = PKL.Pk_plane(np.float32(field), BoxSize, MAS)
        pss.append(Pk2D)
        
    bispecs_in = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_in[j, ii] = bispecs[j+0, 0,0,0][indices[0], indices[1], indices[2]]
            k_values = pss[j].k
            ps_values = pss[j].Pk
            arg_k1_ps = find_nearest_id(k_values, k1)
            arg_k2_ps = find_nearest_id(k_values, k2)
            arg_k3_ps = find_nearest_id(k_values, k)
            ps_values_1 = ps_values[arg_k1_ps]
            ps_values_2 = ps_values[arg_k2_ps]
            ps_values_3 = ps_values[arg_k3_ps]
            bispecs_in[j, ii] = bispecs_in[j, ii]/(ps_values_1*ps_values_2+ps_values_1*ps_values_3+ps_values_2*ps_values_3)
        
    bispecs_out = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_out[j, ii] = bispecs[j+size, 0,0,0][indices[0], indices[1], indices[2]] 
            k_values = pss[j+size].k
            ps_values = pss[j+size].Pk
            arg_k1_ps = find_nearest_id(k_values, k1)
            arg_k2_ps = find_nearest_id(k_values, k2)
            arg_k3_ps = find_nearest_id(k_values, k)
            ps_values_1 = ps_values[arg_k1_ps]
            ps_values_2 = ps_values[arg_k2_ps]
            ps_values_3 = ps_values[arg_k3_ps]
            bispecs_out[j, ii] = bispecs_out[j, ii]/(ps_values_1*ps_values_2+ps_values_1*ps_values_3+ps_values_2*ps_values_3)

    bispecs_pred = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_pred[j, ii] = bispecs[j+2*size, 0,0,0][indices[0], indices[1], indices[2]] 
            k_values = pss[j+2*size].k
            ps_values = pss[j+2*size].Pk
            arg_k1_ps = find_nearest_id(k_values, k1)
            arg_k2_ps = find_nearest_id(k_values, k2)
            arg_k3_ps = find_nearest_id(k_values, k)
            ps_values_1 = ps_values[arg_k1_ps]
            ps_values_2 = ps_values[arg_k2_ps]
            ps_values_3 = ps_values[arg_k3_ps]
            bispecs_pred[j, ii] = bispecs_pred[j, ii]/(ps_values_1*ps_values_2+ps_values_1*ps_values_3+ps_values_2*ps_values_3)

    b_out_mean = np.mean(bispecs_out, axis=0)
    b_pred_mean = np.mean(bispecs_pred, axis=0)
    diff = (b_pred_mean - b_out_mean)/(b_out_mean)
    b_pred_std = np.std(bispecs_pred, axis=0)/np.sqrt(size)
    b_out_std = np.std(bispecs_out, axis=0)/np.sqrt(size)
    den = np.sqrt( (b_pred_std / b_out_mean)**2 + (b_pred_mean * b_out_std / (b_out_mean**2))**2 )
    e = np.mean(np.abs(diff)/den)

    #print('second bispectra')

    k1 = 0.2
    k2 = 0.2
    k3 = np.arange(k2-k1, k2+k1, 0.01)
    theta = np.arccos((k3**2 - k1**2 - k2**2)/(-2*k1*k2))
    theta[0] = 0
    theta[-1]= np.pi
    bin_mid_points = np.arange(0.01, 0.459, 0.01)

    def find_nearest_id(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    arg_k1 = find_nearest_id(bin_mid_points, k1)
    arg_k2 = find_nearest_id(bin_mid_points, k2)

    bispecs_in = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_in[j, ii] = bispecs[j+0, 0,0,0][indices[0], indices[1], indices[2]]

    bispecs_out = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_out[j, ii] = bispecs[j+size, 0,0,0][indices[0], indices[1], indices[2]]

    bispecs_pred = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_pred[j, ii] = bispecs[j+2*size, 0,0,0][indices[0], indices[1], indices[2]]


    b_out_mean = np.mean(bispecs_out, axis=0)
    b_pred_mean = np.mean(bispecs_pred, axis=0)
    diff = (b_pred_mean - b_out_mean)/(b_out_mean)
    b_pred_std = np.std(bispecs_pred, axis=0)/np.sqrt(size)
    b_out_std = np.std(bispecs_out, axis=0)/np.sqrt(size)
    den = np.sqrt( (b_pred_std / b_out_mean)**2 + (b_pred_mean * b_out_std / (b_out_mean**2))**2 )
    f = np.mean(np.abs(diff)/den)

    #print('second bispectra r')

    # reduced bispectrum
    pss = []
    BoxSize = 1000.0 # size of the density field in Mpc/h
    MAS = 'PCS'

    for i in range(3*size):
        field = test_set[i, ..., 0]
        Pk2D = PKL.Pk_plane(np.float32(field), BoxSize, MAS)
        pss.append(Pk2D)
        

    bispecs_in = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_in[j, ii] = bispecs[j+0, 0,0,0][indices[0], indices[1], indices[2]]
            k_values = pss[j].k
            ps_values = pss[j].Pk
            arg_k1_ps = find_nearest_id(k_values, k1)
            arg_k2_ps = find_nearest_id(k_values, k2)
            arg_k3_ps = find_nearest_id(k_values, k)
            ps_values_1 = ps_values[arg_k1_ps]
            ps_values_2 = ps_values[arg_k2_ps]
            ps_values_3 = ps_values[arg_k3_ps]
            bispecs_in[j, ii] = bispecs_in[j, ii]/(ps_values_1*ps_values_2+ps_values_1*ps_values_3+ps_values_2*ps_values_3)

    bispecs_out = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_out[j, ii] = bispecs[j+size, 0,0,0][indices[0], indices[1], indices[2]]
            k_values = pss[j+size].k
            ps_values = pss[j+size].Pk
            arg_k1_ps = find_nearest_id(k_values, k1)
            arg_k2_ps = find_nearest_id(k_values, k2)
            arg_k3_ps = find_nearest_id(k_values, k)
            ps_values_1 = ps_values[arg_k1_ps]
            ps_values_2 = ps_values[arg_k2_ps]
            ps_values_3 = ps_values[arg_k3_ps]
            bispecs_out[j, ii] = bispecs_out[j, ii]/(ps_values_1*ps_values_2+ps_values_1*ps_values_3+ps_values_2*ps_values_3)

    bispecs_pred = np.zeros((size, len(theta)))
    for j in range(size):
        for ii, k in enumerate(k3):
            arg_k3 = find_nearest_id(bin_mid_points, k)
            indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
            bispecs_pred[j, ii] = bispecs[j+2*size, 0,0,0][indices[0], indices[1], indices[2]]
            k_values = pss[j+2*size].k
            ps_values = pss[j+2*size].Pk
            arg_k1_ps = find_nearest_id(k_values, k1)
            arg_k2_ps = find_nearest_id(k_values, k2)
            arg_k3_ps = find_nearest_id(k_values, k)
            ps_values_1 = ps_values[arg_k1_ps]
            ps_values_2 = ps_values[arg_k2_ps]
            ps_values_3 = ps_values[arg_k3_ps]
            bispecs_pred[j, ii] = bispecs_pred[j, ii]/(ps_values_1*ps_values_2+ps_values_1*ps_values_3+ps_values_2*ps_values_3)

    b_out_mean = np.mean(bispecs_out, axis=0)
    b_pred_mean = np.mean(bispecs_pred, axis=0)
    diff = (b_pred_mean - b_out_mean)/(b_out_mean)
    b_pred_std = np.std(bispecs_pred, axis=0)/np.sqrt(size)
    b_out_std = np.std(bispecs_out, axis=0)/np.sqrt(size)
    den = np.sqrt( (b_pred_std / b_out_mean)**2 + (b_pred_mean * b_out_std / (b_out_mean**2))**2 )
    g = np.mean(np.abs(diff)/den)
    
    print(a,b,c,d,e,f,g)
    return a+b+c+d+e+f+g
