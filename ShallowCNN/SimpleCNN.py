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
from numpy import newaxis

class SimpleCNN(object):
    def __init__(self, sess, image_size_H=22, image_size_W=1000,
                 batch_size=50, sample_size=50, y_dim=4, 
                 c_dim=1, dropout=0.9, reg = 0.00, learning_rate = 0.001,
                 checkpoint_dir=None, sample_dir=None):
        #######################################################
        # Initialization of Simple CNN Net
        # Parameters:
        #######################################################
        self.sess = sess
        self.batch_size = batch_size
        self.input_size_W = image_size_W
        self.input_size_H = image_size_H
        self.sample_size = sample_size

        self.reg = reg
        self.lr = learning_rate
        self.dropout = dropout
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.df_dim_1 = 64
        self.df_dim_2 = 64

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def discriminator(self, image, y, reuse=False, train=True):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            #######################################################
            # Define SimpleCNN network structure
            # Adopt: Conv layer / batch norm / l-ReLU(leaky ReLU)
            # Does not contain: maxpooling layer
            #######################################################
            if reuse:
                scope.reuse_variables()

            h1 = lrelu(batch_norm(conv2d(image, self.df_dim_1,stddev=0.01, name='d_h1_conv'),name='d_bn1'))
            h1 = tf.nn.dropout(h1,self.dropout) if (self.dropout > 0  and self.dropout != 1) else h1

            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim_1, stddev=0.01, name='d_h2_conv'),name='d_bn2'))
            h2 = tf.nn.dropout(h2,self.dropout) if (self.dropout > 0  and self.dropout != 1) else h2

            #h2 = tf.layers.max_pooling2d(h1, 5, 5)
            h2 = tf.reshape(h2, [self.batch_size, -1])
            h3 = lrelu(batch_norm(linear(h2, 1024, scope='en_fc3'), train=train, name='d_bn3'))
            h4 = linear(tf.reshape(h2, [self.batch_size, -1]), self.y_dim, 'd_h3_lin')
            # return logits
            return h4

    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of Simple CNN.
        #######################################################
        image_dims = [self.input_size_H, self.input_size_W, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.y = tf.placeholder(tf.int32, [self.batch_size], name='y')

        y_logits = self.discriminator(self.inputs, self.y)
        self.y_pred = tf.nn.softmax(y_logits)
        var   = tf.trainable_variables() 
        
        # TODO: add all W variables into L2-loss 
        self.L2_loss = 0
        y_oh = tf.one_hot(self.y, self.y_dim)
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_oh,logits=self.y_pred)) + self.L2_loss

        self.saver = tf.train.Saver()

    def train(self, Data, config, params=None, debug=True):
        """Train VAE"""
        logging.basicConfig(filename=config.log_file,format='%(message)s',level=logging.DEBUG)

        X_train, y_train = Data.train
        X_test, y_test = Data.test

        # pre-normalize
        
        self.mu = np.mean(X_train, axis=(0,1,2)) # 2[channel] vector
        self.var = np.var(X_train, axis=(0,1,2)) # 
        X_train = (X_train - self.mu) / np.sqrt(self.var + 1e-8)
        

        optim = tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1).minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        start_time = time.time()
        counter = 1

        ### set model and sample folder ###
        sub_dir = "SimpleCNN" # TODO: add parameters
        self.model_dir = os.path.join(self.checkpoint_dir,sub_dir)

        #could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        '''
        if self.load(self.model_dir):
            print("[*] ["+self.model_dir+"] Load SUCCESS")
        else:
            print(" [!]["+self.model_dir+"] Load failed...")
        '''
        self.sample_dir = os.path.join(config.sample_dir,sub_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

        print( " === ["+ sub_dir + "} Start SimpleCNN Training! ===")

        train_accu = 0
        for epoch in xrange(config.epoch): # config.epoch
            batch_idxs = min(X_train.shape[0], config.train_size) // config.batch_size # through all training samples
            #batch_idxs = 1 # fix idx with random selected samples
            for idx in xrange(0, batch_idxs):
                counter += 1
                batch_images = X_train[idx*config.batch_size:(idx+1)*config.batch_size, :] # through all training samples
                batch_labels = y_train[idx*config.batch_size:(idx+1)*config.batch_size]
                #batch_images = sdata[np.random.choice(sdata.shape[0], self.batch_size, replace=False),:] # random select batch_size

                _, loss, y_pred = self.sess.run([optim, self.loss, self.y_pred], feed_dict = {self.inputs:batch_images, self.y: batch_labels})

                y_oh = tf.one_hot(batch_labels, self.y_dim)
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_oh, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Validation Accuracy:", accuracy.eval({self.inputs: batch_images,self.y: batch_labels}))
                logging.info("Validation Accuracy:"+str(accuracy.eval({self.inputs: batch_images,self.y: batch_labels})))
                if accuracy.eval({self.inputs: batch_images,self.y: batch_labels}) > train_accu:
                    train_accu = accuracy.eval({self.inputs: batch_images,self.y: batch_labels})
                #print(binary_cross_entropy(d,batch_images))
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" %(epoch, idx, batch_idxs,time.time() - start_time, loss))
                logging.info("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" %(epoch, idx, batch_idxs,time.time() - start_time, loss))

                if (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(self.model_dir, counter)
                    #pass
        print("Best training accuracy:",train_accu)

    def test(self, Data, config):
        '''
        if self.load(self.model_dir):
            print("[*] ["+self.model_dir+"] Load SUCCESS")
        else:
            print(" [!]["+self.model_dir+"] Load failed...")
        '''
        print( " ===  Start SimpleCNN Training! ===")
        X_train, y_train = Data.train
        X_test, y_test = Data.test
        X_test = (X_test - self.mu) / np.sqrt(self.var + 1e-8)

        y_oh = tf.one_hot(y_test[0:self.sample_size], self.y_dim)
        _, loss, y_pred = self.sess.run([self.discriminator(self.inputs, self.y,reuse=True,train=False), self.loss, self.y_pred],
            feed_dict={self.inputs: X_test[0:self.sample_size],self.y: y_test[0:self.sample_size]})
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_oh, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Test Accuracy:", accuracy.eval({self.inputs: X_test[0:self.sample_size],self.y: y_test[0:self.sample_size]}))
        logging.info("Accuracy:"+str(accuracy.eval({self.inputs: X_test[0:self.sample_size],self.y: y_test[0:self.sample_size]})))

    def save(self, checkpoint_dir, step):
        model_name = "SimpleCNN"
        model_dir = "%s" % ("EEG_frame")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name))

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s" % ("EEG_frame")
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
