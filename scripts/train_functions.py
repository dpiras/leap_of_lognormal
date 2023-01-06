import os
import time
import numpy as np
import tensorflow as tf
from models import generator_loss, discriminator_loss

@tf.function
def train_step_gen(input_image, target, epoch, generator=None, discriminator=None, 
                   generator_optimizer=None):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(gen_output, training=True)
        gen_total_loss, gen_gan_loss, gen_l2_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss, wgan_loss, gp = discriminator_loss(disc_real_output, disc_generated_output, target, gen_output, discriminator)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))

# only if you want to use e.g. tensorboard to keep track of metrics
#  with summary_writer.as_default():
#    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
#    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
#    tf.summary.scalar('gen_l2_loss', gen_l2_loss, step=epoch)
#    tf.summary.scalar('disc_loss', disc_loss, step=epoch)
#    tf.summary.scalar('wgan_loss', wgan_loss, step=epoch)
#    tf.summary.scalar('gp', gp, step=epoch)


@tf.function
def train_step_dis(input_image, target, epoch, generator=None, discriminator=None,
                   discriminator_optimizer=None):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(gen_output, training=True)
        gen_total_loss, gen_gan_loss, gen_l2_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss, wgan_loss, gp = discriminator_loss(disc_real_output, disc_generated_output, target, gen_output, discriminator)

    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

# only if you want to use e.g. tensorboard to keep track of metrics
#  with summary_writer.as_default():
#    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
#    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
#    tf.summary.scalar('gen_l2_loss', gen_l2_loss, step=epoch)
#    tf.summary.scalar('disc_loss', disc_loss, step=epoch)
#    tf.summary.scalar('wgan_loss', wgan_loss, step=epoch)
#    tf.summary.scalar('gp', gp, step=epoch)

def fit(train_ds, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, checkpoint_prefix):
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)
        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 1 == 0:
                print(n.numpy()+1)
            for _ in range(10):
                train_step_dis(input_image, target, epoch, generator=generator, discriminator=discriminator, discriminator_optimizer=discriminator_optimizer)
            train_step_gen(input_image, target, epoch, generator=generator, discriminator=discriminator, generator_optimizer=generator_optimizer)
        print()

        checkpoint.save(file_prefix=checkpoint_prefix+f'/{epoch}/')

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)
