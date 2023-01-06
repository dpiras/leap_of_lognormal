import os
import time
import numpy as np
import tensorflow as tf


def log_transform(x, y):
  return tf.math.log(x+1), tf.math.log(y+1)

def load(load_file):
  # first apply an operation, then return one input and output image
  load_file = load_file.map(log_transform)

  extract_from_dataset = np.array(list(load_file.take(3).as_numpy_iterator()))

  real_image = extract_from_dataset[2, 1:]
  input_image =  extract_from_dataset[2, :1]

  input_image = tf.cast(input_image, tf.float64)
  real_image = tf.cast(real_image, tf.float64)

  return input_image[0, :, :], real_image[0, :, :]


def periodic_padding(image, padding=1):
    '''
    Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
    Note padding has to be >0!
    '''

    if padding == 0:
      return image
    #print(image.shape)

    upper_pad = image[:, -padding:,:]
    lower_pad = image[:, :padding,:]

    partial_image = tf.concat([upper_pad, image, lower_pad], axis=1)

    #print(partial_image.shape)

    left_pad = partial_image[:, :,-padding:]
    right_pad = partial_image[:, :,:padding]

    padded_image = tf.concat([left_pad, partial_image, right_pad], axis=2)

    #print(padded_image.shape)

    return padded_image

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

 
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='valid',
                             kernel_initializer=initializer, 
                             use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='valid',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.3))

  result.add(tf.keras.layers.ReLU())

  return result

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i][..., 0])
    plt.colorbar()
    plt.axis('off')
  plt.subplot(2, 3, 4)
  plt.title('Input-output')
  plt.scatter(test_input[0].numpy().flatten(), tar[0].numpy().flatten())
  plt.subplot(2, 3, 5)
  plt.title('Input-prediction')
  plt.scatter(test_input[0].numpy().flatten(), prediction[0].numpy().flatten())
  plt.subplot(2, 3, 6)
  plt.title('Output-prediction')
  plt.scatter(tar[0].numpy().flatten(), prediction[0].numpy().flatten())
  print( 'Input vs output:', ((test_input.numpy()-tar.numpy())**2).mean(), '. Input vs prediction:', ((test_input.numpy()-prediction.numpy())**2).mean(), '. Output vs prediction:', ((prediction.numpy()-tar.numpy())**2).mean())
  plt.show()
