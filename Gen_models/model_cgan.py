from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
import logging

class CGAN(object):
    def __init__(self, sess, image_size_H=26, image_size_W=41,
                 batch_size=64, sample_size=100, output_size_H=26, y_dim = 4,
                 output_size_W=41, z_dim=100, c_dim=2, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """
        Initialization of Conditional GAN
        Paramters: see CGAN-Trainer.py
        """
        self.sess = sess
        self.batch_size = batch_size
        self.input_size_W = image_size_W
        self.input_size_H = image_size_H
        self.sample_size = sample_size
        self.output_size_W = output_size_W
        self.output_size_H = output_size_H

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.y_dim = y_dim

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        # layer dimension
        self.gf_dim = 64
        self.df_dim = 64
        self.gfc_dim = 1024
        self.dfc_dim = 1024
        self.build_model()

    def discriminator(self, image, y, reuse=False, train=True):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            #######################################################
            # Discriminator: ConvNet
            # Structure: 
            # Input[26,41,2]+Label[4] -> [Conv] -> [BN] -> [lReLU] -> [Conv] -> [BN] -> [lReLU] -> [FC] -> [Sigmoid] -> Output[1]
            #######################################################
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            images = conv_cond_concat(image,yb)

            h1 = lrelu(batch_norm(conv2d(images, self.df_dim, name='d_h1_conv'),name='d_bn1'))
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*2, name='d_h2_conv'),name='d_bn2'))
            h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3


    def generator(self, z, y, reuse=False, train=True):
        with tf.variable_scope("generator", reuse=reuse) as scope:
            #######################################################
            # Generator: ConvNet
            # Structure:
            # Input[z]+Label[4] -> [FC] -> [BN] -> [ReLU] -> [DeConv] -> [BN] -> [ReLU] -> [DeConv] -> [tanh] -> Output[26,41,2]
            # When not in train mode, it works as sampler (spectrum generator)
            #######################################################
            if reuse:
                scope.reuse_variables()

            tf.to_float(y, name='ToFloat')
            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            # define generator network
            if train:
                s_h, s_w = self.output_size_H, self.output_size_W
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*2*s_h4*s_w4, 'g_h0_lin', with_w=True)
                self.h1 = tf.reshape(self.z_, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = tf.nn.relu(batch_norm(self.h1,name='g_bn1'))
                h1 = conv_cond_concat(h1,yb)

                h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h1', with_w=True)
                h2 = tf.nn.relu(batch_norm(h2,name='g_bn2'))
                h2 = conv_cond_concat(h2,yb)
        
                h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h2', with_w=True)

                return tf.nn.sigmoid(h3)
            else:
                # sampling images using trained model (reuse=True)
                s_h, s_w = self.output_size_H, self.output_size_W
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

                h1 = linear(z, self.gf_dim*2*s_h4*s_w4, 'g_h0_lin')
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = tf.nn.relu(batch_norm(h1,train=False,name='g_bn1'))
                h1 = conv_cond_concat(h1,yb)

                h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h1')
                h2 = tf.nn.relu(batch_norm(h2,train=False,name='g_bn2'))
                h2 = conv_cond_concat(h2,yb)

                h3 = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h2')

                return tf.nn.sigmoid(h3)


    def build_model(self):
        #######################################################
        # Define inputs, operations on inputs and loss of DCGAN.
        # Discriminator loss has two parts: cross entropy for real
        # images and cross entropy for fake images generated by 
        # generator.
        #######################################################
        image_dims = [self.input_size_H, self.input_size_W, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='z')

        # generate and discrimate
        self.G = self.generator(self.z, self.y)
        self.Dr, self.Dr_logits = self.discriminator(inputs, self.y, reuse=False)
        self.Df, self.Df_logits = self.discriminator(self.G, self.y, reuse=True) #setting reuse=True
        '''
        self.d_rsum = histogram_summary("d", self.Dr)
        self.d__sum = histogram_summary("d_", self.Df)
        self.G_sum = image_summary("G", self.G)
        '''
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dr_logits, labels=tf.ones_like(self.Dr)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Df_logits, labels=tf.zeros_like(self.Df)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Df_logits, labels=tf.ones_like(self.Df)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)                      
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
        #######################################################
        #                   end of your code
        #######################################################
        # define var lists for generator and discriminator
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, Data, data_tuple, config, params, mask, debug=True):
        # create two optimizers for generator and discriminator,
        # and only update the corresponding variables.
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        logging.basicConfig(filename=config.log_file,format='%(message)s',level=logging.DEBUG)

        sdata, sdata_labels = data_tuple
        # pre-normalize
        mu = np.mean(sdata, axis=(0,1,2)) # 2[channel] vector
        var = np.var(sdata, axis=(0,1,2)) # 
        sdata = (sdata - mu) / np.sqrt(var + 1e-8)

        start_time = time.time()
        counter = 1

        sub_dir = "e"+str(params['electrode_id'])
        model_dir = os.path.join(self.checkpoint_dir,sub_dir)

        sample_dir = os.path.join(config.sample_dir,sub_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        print( " === ["+ sub_dir + "} Start C-GAN Training! ===")

        if config.train:
            for epoch in xrange(config.epoch): #config.epoch
                batch_idxs = min(sdata.shape[0], config.train_size) // config.batch_size
                #batch_idxs = 30
                for idx in xrange(0, batch_idxs):
                    batch_images = sdata[idx * config.batch_size:(idx + 1) * config.batch_size, :]
                    batch_labels = sdata_labels[idx * config.batch_size:(idx + 1) * config.batch_size, :]
                     #inputs
                    #batch_images = sdata[np.random.choice(sdata.shape[0], self.batch_size, replace=False),:]
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32) #z

                    # update D network (change d_sum)
                    self.sess.run([d_optim],feed_dict={self.inputs: batch_images,self.z: batch_z, self.y:batch_labels})
                    # update G network(twice) (change g_sum)
                    self.sess.run([g_optim],feed_dict={self.z: batch_z, self.y:batch_labels})
                    self.sess.run([g_optim],feed_dict={self.z: batch_z, self.y:batch_labels})

                    # conpute error in 3 part
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
                    errD_real = self.d_loss_real.eval({self.inputs: batch_images, self.y:batch_labels})
                    errG = self.g_loss.eval({self.z: batch_z, self.y:batch_labels})

                    counter += 1

                    logging.info("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f | counter:[%4d]" %\
                        (epoch, idx, batch_idxs,time.time() - start_time, errD_fake+errD_real, errG, counter))
                    
                    if (idx == batch_idxs-1) and debug == True:# test when training (defining sampler)
                        samples, d_loss, g_loss = self.sess.run([self.generator(self.z,self.y, reuse=True,train=False), self.d_loss, self.g_loss],
                            feed_dict={self.z: batch_z,self.inputs: batch_images, self.y:batch_labels})
                        save_images(samples[0,:,:,0].reshape((1,)+samples[0,:,:,0].shape+(1,)), image_manifold_size(1),'./{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
                        print("[Sample training] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                        logging.info("[Sample training] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    if np.mod(counter, 50) == 1 or (idx == batch_idxs-1):# save model
                        self.save(model_dir, counter)

            # generator (sampler after training)
            print( " === [" + sub_dir + "} Start C-GAN Sampling! ===")
            sample_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            sample_images = sdata[mask,:]
            sample_y = sdata_labels[mask,:]

            samples, d_loss, g_loss = self.sess.run([self.generator(self.z, self.y, reuse=True,train=False), self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z,self.inputs: sample_images, self.y: sample_y})
            save_images(samples[0,:,:,0].reshape((1,)+samples[0,:,:,0].shape+(1,)), image_manifold_size(1),'./{}/test.png'.format(sample_dir))
            print("[Sample test %s ] d_loss: %.8f, g_loss: %.8f" % (sub_dir, d_loss, g_loss))
            logging.info("[Sample test %s ] d_loss: %.8f, g_loss: %.8f" % (sub_dir, d_loss, g_loss))

            # post-processing
            samples = (samples - 0.5) * 2.0 # sample to [-1,1]
            samples = samples * np.sqrt(var + 1e-8) + mu ## broadcasting  add norm
            sample_labels = np.zeros((self.batch_size))
            for i in range(self.batch_size):
                sample_labels[i] = 769 + np.where(sample_y[i] == 1)[0]
            return samples


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format("EEG_frame", self.batch_size,self.output_size_H, self.output_size_W)

    def save(self, checkpoint_dir, step):
        model_name = "CGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
