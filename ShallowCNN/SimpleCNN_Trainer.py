import os
import scipy.misc
import numpy as np

from SimpleCNN import SimpleCNN
from utils import pp, visualize, to_json
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import data
import logging

flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 0.02, "Uper bound of learning rate for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 50, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 50, "The size of batch images [64]")
flags.DEFINE_integer("image_size_W", 1000, "The size of image to use (will be center cropped) [W]")
flags.DEFINE_integer("image_size_H", 22, "The size of image to use (will be center cropped) [H]")
flags.DEFINE_integer("y_dim", 4, "Dimension of image color. [2]")
flags.DEFINE_integer("c_dim", 1, "Dimension of latent code. [100]")
flags.DEFINE_string("checkpoint_dir", "./models/SimpleCNN", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples/SimpleCNN", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_file",'SimpleCNN.log',"record log file[./model/vae1.log]")
flags.DEFINE_boolean("is_train", True, "True for tcd raining, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    Data = data.Data()
    Data.load_data()

    with tf.Session() as sess:
        mode = {'test_obj':None, 'train_obj':None, 'test_size':None, 'train_size':None}
        Data.train_test_split(mode,timesize=100)
        print(Data.train[0].shape)
        scnn = SimpleCNN(sess,
            image_size_W=Data.timeWindow,
            image_size_H=FLAGS.image_size_H,
            batch_size=FLAGS.batch_size,
            c_dim=FLAGS.c_dim,
            y_dim=FLAGS.y_dim,
            learning_rate = FLAGS.learning_rate,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            sample_size=FLAGS.sample_size)

        scnn.train(Data, FLAGS)
        scnn.test(Data, FLAGS)

    tf.reset_default_graph() 

if __name__ == '__main__':
    tf.app.run()
