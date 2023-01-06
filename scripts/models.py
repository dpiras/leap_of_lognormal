import os
import time
import numpy as np
import tensorflow as tf
from utils import downsample, upsample, periodic_padding

IMG_WIDTH = 128
OUTPUT_CHANNELS = 1
LAMBDA = 10.0 # lambda for MSE
lambda_ = 100.0  # this is the lambda for the gradient penalty

def Generator():
  inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
    downsample(128, 4),  # (bs, 64, 64, 128)
    downsample(256, 4),  # (bs, 32, 32, 256)
    downsample(512, 4),  # (bs, 16, 16, 512)
    downsample(512, 4),  # (bs, 8, 8, 512)
    downsample(512, 4),  # (bs, 4, 4, 512)
    downsample(512, 4),  # (bs, 2, 2, 512)
  ]

  up_stack = [
    upsample(512, 2, apply_dropout=True),  # (bs, 2, 2, 1024)
    upsample(512, 2, apply_dropout=True),  # (bs, 4, 4, 1024)
    upsample(512, 2, apply_dropout=True),  # (bs, 8, 8, 1024)
    upsample(512, 2),  # (bs, 16, 16, 1024)
    upsample(256, 2),  # (bs, 32, 32, 512)
    upsample(128, 2),  # (bs, 64, 64, 256)
    upsample(64, 2),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         #kernel_initializer=initializer,
                                         #activation='tanh'
                                         )  # (bs, 128, 128, 1)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = periodic_padding(x, padding=1)
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x1_shape = tf.shape(skip)
    x2_shape = tf.shape(x)

    height_diff = (x1_shape[1] - x2_shape[1]) // 2
    width_diff = (x1_shape[2] - x2_shape[2]) // 2

    skip_cropped = skip[:,
                                    height_diff: (x2_shape[1] + height_diff),
                                    width_diff: (x2_shape[2] + width_diff),
                                    :]
    x = tf.keras.layers.Concatenate()([x, skip_cropped])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = -tf.reduce_mean(disc_generated_output)
  #loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  target = tf.cast(target, tf.float32)
  # mean absolute error
  l2_loss = tf.reduce_mean(tf.square(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l2_loss)

  return total_gen_loss, gan_loss, l2_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_WIDTH, 1], name='input_image')

  x = inp

  x = periodic_padding(x, padding=1)
  down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
  down1 = periodic_padding(down1, padding=1)
  down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
  down2 = periodic_padding(down2, padding=1)
  down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

  down3 = periodic_padding(down3, padding=1)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                use_bias=False)(down3)  # (bs, 31, 31, 512)

  leaky_relu = tf.keras.layers.LeakyReLU()(conv)

  zero_pad2 = periodic_padding(leaky_relu, padding=1) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                )(zero_pad2)  # (bs, 30, 30, 1)

  last=tf.keras.layers.Flatten()(last)
  last = tf.keras.layers.Dense(1)(last)

  return tf.keras.Model(inputs=inp, outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output, target, gen_output, discriminator):

  wgan_d_loss = tf.reduce_mean(disc_generated_output) - tf.reduce_mean(disc_real_output)

  target = tf.cast(target, tf.float32)

  alpha = tf.random.uniform(shape=[tf.shape(target)[0]], minval=0., maxval=1.)
  alpha = tf.reshape(tf.repeat(alpha, IMG_WIDTH*IMG_WIDTH, axis=0), [-1, IMG_WIDTH, IMG_WIDTH, 1])
  differences = gen_output - target
  interpolates = target + (alpha * differences)
  gradients = tf.gradients(discriminator(interpolates, training=True), [interpolates])[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
  gradient_penalty = tf.reduce_mean((slopes-1.)**2)

  total_disc_loss = wgan_d_loss + lambda_ * gradient_penalty

  return total_disc_loss, wgan_d_loss, gradient_penalty
