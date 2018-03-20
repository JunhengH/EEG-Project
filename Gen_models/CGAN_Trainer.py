import os
import scipy.misc
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from model_cgan import CGAN
from utils import pp, visualize, to_json, show_all_variables
import data
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Epoch to train [30]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [200]")
flags.DEFINE_integer("image_size_W", 41, "The size of image to use (will be center cropped) [W]")
flags.DEFINE_integer("image_size_H", 26, "The size of image to use (will be center cropped) [H]")
flags.DEFINE_integer("c_dim", 2, "Dimension of image color. [2]")
flags.DEFINE_integer("y_dim", 4, "Dimension of image color. [2]")
flags.DEFINE_integer("z_dim", 100, "Dimension of latent code. [100]")
flags.DEFINE_integer("output_size_W", 41, "The size of the output images to produce [W]")
flags.DEFINE_integer("output_size_H", 26, "The size of the output images to produce [H]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "./models/cgan", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples/cgan", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_file",'ecg_cgan.log',"record log file[./model/gan.log]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
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

    sample_container = np.zeros((Data.generate_size,Data.electrode_num, Data.freDim, Data.timeDim, Data.channel))
    quarter_ones = np.ones((Data.generate_size/4))
    sample_label = np.concatenate((quarter_ones*769,quarter_ones*770,quarter_ones*771, quarter_ones*772)) #target labels

    for electrode in range(22):
        params = {'electrode_id':electrode,'label_id':None, 'subject':None}
        sdata, sdata_labels = Data.get_slice(params['electrode_id'],target=params['label_id'],subject=params['subject'])
        # generate mask
        pool = np.zeros(sdata_labels.shape[0])
        for i in range(sdata_labels.shape[0]):
            pool[i] = np.where(sdata_labels[i] == 1)[0]
        mask = np.zeros(Data.generate_size)
        subsize = Data.generate_size/4
        for i in range(4):
            mask[subsize*i:subsize*(i+1)] = np.random.choice(np.where(pool == i)[0], subsize, replace=False)
        mask = np.asarray(mask, dtype=np.int32)

        # given tuple
        data_tuple = (sdata, sdata_labels)
        with tf.Session(config=run_config) as sess:
            cgan = CGAN(
                sess,
                image_size_W=FLAGS.image_size_W,
                image_size_H=FLAGS.image_size_H,
                batch_size=FLAGS.batch_size,
                output_size_W=FLAGS.output_size_W,
                output_size_H=FLAGS.output_size_H,
                sample_size=Data.generate_size,
                c_dim=FLAGS.c_dim,
                z_dim=FLAGS.z_dim,
                y_dim=FLAGS.y_dim,
                dataset_name=FLAGS.dataset,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)
            samples = cgan.train(Data, data_tuple,FLAGS,params, mask)

        tf.reset_default_graph()
        sample_container[:,electrode,:, :, :] = samples

    # save to Data.gen
    print(" Save generated data!")
    Data.load_gen(sample_container,sample_label,2)

if __name__ == '__main__':
    tf.app.run()
