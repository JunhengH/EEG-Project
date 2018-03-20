from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
#from skimage import io

from ops import *
from utils import *
import random
import numpy as np
import logging
import data

class VAE(object):
    def __init__(self, sess, image_size_H=26, image_size_W=41,
                 batch_size=100, sample_size=100, output_size_H=26,
                 output_size_W=41, z_dim=100, c_dim=2, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """
        Initialization of VAE
        Paramters: see GAN-Trainer.py
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
        self.df_dim = 64
        self.gf_dim = 64

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def encoder(self, image, reuse=False, train=True):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            #######################################################
            # Discriminator: ConvNet
            # Structure: 
            # Input[26,41,2] -> [Conv] -> [BN] -> [lReLU] -> [Conv] -> [BN] -> [lReLU] -> [FC] -> [Sigmoid] -> Output[1]
            #######################################################
            if reuse:
                scope.reuse_variables()

            n_latent = self.z_dim
            h1 = lrelu(batch_norm(conv2d(image, self.df_dim, name='d_h1_conv'),name='d_bn1'))
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*2, name='d_h2_conv'),name='d_bn2'))
            h2 = tf.reshape(h2, [self.batch_size, -1])
            h3 = lrelu(batch_norm(linear(h2, 256, scope='en_fc3'), train=train, name='d_bn3'))

            mn = tf.layers.dense(h3, units=n_latent)
            sd = 0.5 * tf.layers.dense(h3, units=n_latent) 

            return mn, sd


    def decoder(self, z, reuse=False, train=True):
        with tf.variable_scope("decoder", reuse=reuse) as scope:
            #######################################################
            # Generator: ConvNet
            # Structure:
            # Input[z] -> [FC] -> [BN] -> [ReLU] -> [DeConv] -> [BN] -> [ReLU] -> [DeConv] -> [tanh] -> Output[26,41,2]
            # When not in train mode, it works as sampler (spectrum generator)
            #######################################################
            if reuse:
                scope.reuse_variables()

            if train:
                s_h, s_w = self.output_size_H, self.output_size_W
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*2*s_h4*s_w4, 'g_h0_lin', with_w=True)
                self.h1 = tf.reshape(self.z_, [-1, s_h4, s_w4, self.gf_dim * 2])
                h1 = tf.nn.relu(batch_norm(self.h1,name='g_bn1'))

                h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h1', with_w=True)
                h2 = tf.nn.relu(batch_norm(h2,name='g_bn2'))
        
                h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h2', with_w=True)

                return tf.nn.sigmoid(h3)
            else:
                # sampling images using trained model
                s_h, s_w = self.output_size_H, self.output_size_W
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

                h1 = linear(z, self.gf_dim*2*s_h4*s_w4, 'g_h0_lin')
                h1 = tf.reshape(h1, [-1, s_h4, s_w4, self.gf_dim * 2])
                h1 = tf.nn.relu(batch_norm(h1,train=False,name='g_bn1'))

                h2 = deconv2d(h1, [self.sample_size, s_h2, s_w2, self.gf_dim*1], name='g_h1')
                h2 = tf.nn.relu(batch_norm(h2,train=False,name='g_bn2'))
        
                h3 = deconv2d(h2, [self.sample_size, s_h, s_w, self.c_dim], name='g_h2')

                return tf.nn.sigmoid(h3)

    def build_model(self):
        #######################################################
        # Define inputs, operations on inputs and loss of VAE.
        # Loss term has two parts: reconstruction loss (L2-loss) 
        # and KL divergence loss. 
        # Implement reparameterization trick to sample z.
        #######################################################
        image_dims = [self.input_size_H, self.input_size_W, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        mn, sd = self.encoder(inputs)
        self.z = mn + sd * tf.random_normal(tf.shape(sd), 0, 1, dtype=tf.float32)
        self.mn, self.sd = mn, sd
        dec = self.decoder(self.z)
        self.dec = dec

        inputs_flat = tf.reshape(inputs,[-1, self.input_size_H*self.input_size_W*self.c_dim])
        dec_flat = tf.reshape(dec, [-1, self.input_size_H*self.input_size_W*self.c_dim])
        recon_loss = tf.reduce_sum(tf.squared_difference(inputs_flat, dec_flat), 1)
        kl_loss = 0.5* tf.reduce_sum(tf.square(self.mn) + tf.square(sd)-tf.log(1e-8 + tf.square(sd))-1, [1])

        self.recon_loss = tf.reduce_mean(recon_loss)
        self.kl_loss = tf.reduce_mean(kl_loss)
        self.loss = self.recon_loss + self.kl_loss

        self.saver = tf.train.Saver()

    def train(self, Data, config, params, debug=True):
        """Train VAE"""
        logging.basicConfig(filename=config.log_file,format='%(message)s',level=logging.DEBUG)
        sdata, _ = Data.get_slice(params['electrode_id'],target=params['label_id'],subject=params['subject'])

        # pre-normalize
        mu = np.mean(sdata, axis=(0,1,2)) # 2[channel] vector
        var = np.var(sdata, axis=(0,1,2)) # 
        sdata = (sdata - mu) / np.sqrt(var + 1e-8)

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        start_time = time.time()
        counter = 1

        ### set model and sample folder ###
        sub_dir = "e"+str(params['electrode_id'])+"_t"+str(params['label_id'])
        model_dir = os.path.join(self.checkpoint_dir,sub_dir)

        sample_dir = os.path.join(config.sample_dir,sub_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        print( " === ["+ sub_dir + "} Start VAE Training! ===")

        for epoch in xrange(config.epoch): # config.epoch
            batch_idxs = min(sdata.shape[0], config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                counter += 1
                batch_images = sdata[idx*config.batch_size:(idx+1)*config.batch_size, :]
                batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                _, ls, d, i_ls, d_ls, _, _ = self.sess.run([optim, self.loss, self.dec, self.recon_loss, self.kl_loss, self.mn, self.sd], feed_dict = {self.inputs:batch_images})

                error_recon = self.recon_loss.eval({self.z: batch_z,self.inputs:batch_images})
                error_latent = self.kl_loss.eval({self.z: batch_z,self.inputs:batch_images})
                error_total = self.loss.eval({self.z: batch_z,self.inputs:batch_images})
                
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, recon_loss: %.6f, kl_loss: %.6f, loss: %.6f | counter:[%4d]" %\
                    (epoch, idx, batch_idxs,time.time() - start_time, error_recon, error_latent, error_total, counter))
                logging.info("Epoch: [%2d] [%4d/%4d] time: %4.4f, recon_loss: %.6f, kl_loss: %.6f, loss: %.6f | counter:[%4d]" %\
                    (epoch, idx, batch_idxs,time.time() - start_time, error_recon, error_latent, error_total, counter))

                # training sampling
                if np.mod(idx, 7) == 0 and debug == True:# test when training (defining sampler)
                    samples, recon_loss, kl_loss, loss = self.sess.run([self.decoder(self.z,reuse=True,train=False), self.recon_loss, self.kl_loss, self.loss],
                        feed_dict={self.z: batch_z,self.inputs: batch_images})
                    save_images(samples[0,:,:,0].reshape((1,)+samples[0,:,:,0].shape+(1,)), image_manifold_size(1),'./{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
                    print("[Sample train %s ] recon_loss: %.6f, kl_loss: %.6f, loss: %.6f" % (sub_dir, recon_loss, kl_loss, loss))
                    logging.info("[Sample train %s ] recon_loss: %.6f, kl_loss: %.6f, loss: %.6f" % (sub_dir, recon_loss, kl_loss, loss))
                #######################################################
                #                   end of your code
                #######################################################
                if np.mod(counter, 50) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(model_dir, counter)

        # generator (sampler after training)
        print( " === [" +sub_dir + "} Start VAE Sampling! ===")
        sample_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        sample_images = sdata[np.random.choice(sdata.shape[0], self.batch_size, replace=False),:]
        samples, recon_loss, kl_loss, loss = self.sess.run([self.decoder(self.z,reuse=True,train=False), self.recon_loss, self.kl_loss, self.loss],
                        feed_dict={self.z: sample_z, self.inputs: sample_images})
        save_images(samples[0,:,:,0].reshape((1,)+samples[0,:,:,0].shape+(1,)), image_manifold_size(1),'./{}/test.png'.format(sample_dir))
        print("[Sample test %s ] recon_loss: %.6f, kl_loss: %.6f, loss: %.6f" % (sub_dir, recon_loss, kl_loss, loss))
        logging.info("[Sample test %s ] recon_loss: %.6f, kl_loss: %.6f, loss: %.6f" % (sub_dir, recon_loss, kl_loss, loss))
        # post-processing
        samples = (samples - 0.5) * 2.0 # sample to [-1,1]
        samples = samples * np.sqrt(var + 1e-8) + mu ## broadcasting  add norm
        return samples

    def test(self, config):
        pass

    def save(self, checkpoint_dir, step):
        model_name = "vae_model"
        model_dir = "%s_%s_%s_%s" % ("EEG_frame", self.batch_size, self.output_size_H, self.output_size_W)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s_%s_%s" % ("EEG_frame", self.batch_size, self.output_size_H, self.output_size_W)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
