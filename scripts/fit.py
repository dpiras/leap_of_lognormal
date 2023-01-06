"""
Script to train the WGAN-GP model on the pairs of highly-correlated lognormal and N-body slices. 
Code is not extremely factorised, and requires the training data to be obtained somewhere else.
Uses the functions in models.py, utils.py and train_functions.py.
Get in touch with dr.davide.piras@gmail.com if you need help with this script.
"""

import os
import time
import numpy as np
import tensorflow as tf
from utils import log_transform, periodic_padding, downsample, upsample
from train_functions import fit
from models import Generator, Discriminator, generator_loss, discriminator_loss

BUFFER_SIZE = 600 
BATCH_SIZE = 32 
IMG_WIDTH = 128 
IMG_HEIGHT = IMG_WIDTH 

data_path = 'data_path/'
file_name = '/tf_dataset/'

# load dataset; only train here
load_tf_dataset_train = tf.data.experimental.load(path=data_path+f'train_dataset', compression='GZIP',
                              element_spec=(tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float64),
                              tf.TensorSpec(shape=(IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float64)))

# filters to remove nans and infs
filter_nan = lambda x, y: not tf.reduce_any(tf.math.is_nan(x)) and not tf.reduce_any(tf.math.is_nan(y))

train_dataset = load_tf_dataset_train
train_dataset = train_dataset.filter(filter_nan).map(log_transform)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# architecture
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.0, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.0, beta_2=0.9)

checkpoint_dir = data_path+'/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
EPOCHS = 1000

fit(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix)

