import os
import scipy.misc
import numpy as np

from model_vae import VAE
from utils import pp, visualize, to_json
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import data
import logging

flags = tf.app.flags
flags.DEFINE_integer("epoch", 8, "Epoch to train [50]")
flags.DEFINE_float("learning_rate", 0.0002, "Uper bound of learning rate for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [64]")
flags.DEFINE_integer("image_size_W", 41, "The size of image to use (will be center cropped) [W]")
flags.DEFINE_integer("image_size_H", 26, "The size of image to use (will be center cropped) [H]")
flags.DEFINE_integer("output_size_W", 41, "The size of the output images to produce [W]")
flags.DEFINE_integer("output_size_H", 26, "The size of the output images to produce [H]")
flags.DEFINE_integer("c_dim", 2, "Dimension of image color. [2]")
flags.DEFINE_integer("z_dim", 100, "Dimension of latent code. [100]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "./models/vae", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples/vae", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_file",'ecg_vae.log',"record log file[./model/vae1.log]")
flags.DEFINE_boolean("is_train", True, "True for tcd raining, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
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

  sample_set = []
  sample_label = []
  for label in range(769,773):
    sample_container = np.zeros((Data.generate_size,Data.electrode_num, Data.freDim, Data.timeDim, Data.channel))
    for electrode in range(22):
      params = {'electrode_id':electrode,'label_id':label, 'subject':None}
      with tf.Session() as sess:
        vae = VAE(sess,
          image_size_W=FLAGS.image_size_W,
          image_size_H=FLAGS.image_size_H,
          batch_size=FLAGS.batch_size,
          output_size_W=FLAGS.output_size_W,
          output_size_H=FLAGS.output_size_H,
          c_dim=FLAGS.c_dim,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          sample_size=Data.generate_size)

        samples = vae.train(Data,FLAGS,params)
      tf.reset_default_graph()
      sample_container[:,electrode,:, :, :] = samples
    
    if len(sample_set) == 0 :
      sample_set = sample_container
      sample_label = np.ones((Data.generate_size)) * label
    else:
      # concatenate
      sample_set = np.concatenate((sample_set, sample_container), axis=0)
      sample_label = np.concatenate((sample_label, np.ones((Data.generate_size)) * label))
    
  # save to Data.gen
  print(" Save generated data!")
  select_mask = np.random.choice(sample_set.shape[0], Data.generate_size, replace=False)
  sample_set = sample_set[select_mask]
  sample_label = sample_label[select_mask]
  Data.load_gen(sample_set,sample_label,0)

if __name__ == '__main__':
    tf.app.run()
