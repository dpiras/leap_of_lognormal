"""
Script to take test images from the best model. This script does NOT calcualate the summary statistics
and the figures that then went intp the paper; to access these, ask Davide Piras.
Code is not extremely factorised, and requires other scripts in this folder.
Uses the functions in models.py and test_single_epoch.py.
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

n_test = 100
BUFFER_SIZE = 600
BATCH_SIZE = 32
IMG_WIDTH = 128
IMG_HEIGHT = IMG_WIDTH

data_path = 'data_path/'
file_name = 'tf_dataset'

test_set_ = tf.data.experimental.load(path=data_path+file_name, compression='GZIP',
                          element_spec=(tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float64),
                          tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float64)))

test_set = np.array(list(test_set_.take(-1).as_numpy_iterator()))[:n_test]

# architecture
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.0, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.0, beta_2=0.9)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
best_epoch = 122 # best so far

checkpoint.restore(tf.train.latest_checkpoint(data_path+f'datasets/training_checkpoints/ckpt/{best_epoch}/'))

current_val_set = np.empty((3*n_test, IMG_WIDTH, IMG_WIDTH, 1))

# take lognormal images
for i in range(n_test):
    current_val_set[i, :, :, 0] = test_set[i, 0, :, :, 0]
# take N-body images
for i in range(n_test):
    current_val_set[i+n_test, :, :, 0] = test_set[i, 1, :, :, 0]
 
# map lognormal through model
for i in range(n_test):
    temp = current_val_set[i:i+1]
    # need to transform to feed into the generator
    temp = np.log(temp+1)
    current_val_set[2*n_test+i, :, :, 0] = np.exp(generator(temp, training=True)[0, ..., 0])-1

# save array
np.save(data_path+'128_results', current_val_set)

# then we separately evaluate the performance of this model,
# and plot the figures that go in the paper
