"""
Script to validate and find the best model among those saved at each epoch.
Code is not extremely factorised, and imports function from test_single_epoch.
Get in touch with dr.davide.piras@gmail.com if you need help with this script.
"""

import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import tensorflow as tf

from models import Generator, Discriminator, generator_loss, discriminator_loss
from test_single_epoch import test_single_model

n_v = 19
BUFFER_SIZE = 600
BATCH_SIZE = 32
IMG_WIDTH = 128
IMG_HEIGHT = IMG_WIDTH

data_path = 'data_path'
file_name = 'where_you_saved_your_validation_dataset'

val_set_ = tf.data.experimental.load(path=data_path+file_name, compression='GZIP',
                          element_spec=(tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float64),
                          tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float64)))
val_set = np.array(list(val_set_.take(-1).as_numpy_iterator())) # (19*700=13300, 2, 128, 128, 1)


# architecture
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.0, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.0, beta_2=0.9)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 150 # total number of epochs for which you trained your model
final_perfs = []
res_arr = np.zeros((EPOCHS, 2))

for e in range(EPOCHS):
    mean_perfs = [] # to keep track of current model performance
    # where you stored the trained models at each epoch
    checkpoint.restore(tf.train.latest_checkpoint(data_path+f'datasets/training_checkpoints/ckpt/{e}/'))

    for b in range(20): # 20 is just to avoid memory overload
        current_val_set_ = val_set[b*n_v:(b+1)*n_v] # (19, 2, 128, 128, 1)
        nan_idx = np.unique(np.argwhere(np.isnan(current_val_set_))[:, 0])
        if len(nan_idx) > 0:
            mask = np.array([True]*n_v)
            mask[nan_idx] = False
            current_val_set_ = current_val_set_[mask]
        new_n_v = current_val_set_.shape[0] # change in case you removed any nans

        current_val_set = np.empty((new_n_v*3, 128, 128, 1))

        # take lognormal images
        for i in range(new_n_v):
            current_val_set[i, :, :, 0] = current_val_set_[i, 0, :, :, 0]
        # take N-body images
        for i in range(new_n_v):
            current_val_set[i+new_n_v, :, :, 0] = current_val_set_[i, 1, :, :, 0]
         
        # map lognormal through model
        for i in range(new_n_v):
            temp = current_val_set[i:i+1]
            # need to transform to feed into the generator
            temp = np.log(temp+1)
            current_val_set[2*new_n_v+i, :, :, 0] = np.exp(generator(temp, training=True)[0, ..., 0])-1

        # then here we evaluate the performance over summary statistics
        mean_perf = test_single_model(current_val_set, new_n_v)
        mean_perfs.append(mean_perf)

    mean_perfs = np.array(mean_perfs)
    mean_perfs = mean_perfs[np.logical_not(np.isnan(mean_perfs))] # remove any possible nan
    final_perf = np.mean(mean_perfs)
    print(e, final_perf)
    res_arr[e] = np.array([e, final_perf])
    np.save(data_path+'val_results', res_arr)
    final_perfs.append(final_perf)

print(f'Best model: {np.argmin(final_perfs)}')


