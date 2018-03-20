#train.py
import data_helpers
from rnn import RNN
from scipy import spatial
from sklearn.metrics import *
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pickle as cPickle
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import yaml, time, datetime
import os, sys, random
try:
	import ujson as json
except:
	print('Cannot import ujson, import json instead.', file=sys.stderr)
	import json
try:
	from smart_open import smart_open as open
except:
	print('smart_open inexists, use the original open instead.', file=sys.stderr)



def handle_flags():

	tf.flags.DEFINE_string("train_data_set", '-1', "Index of training data set used(default:-1, use all)")
	tf.flags.DEFINE_string("test_data_set", '-1', "Index of test and valid data set used(default:-1, use all)")
	tf.flags.DEFINE_integer("start_time", 0, "start time of dataset being used, default:0")
	tf.flags.DEFINE_integer("sequence_length", 1000, "length of data being used, default:1000(all)")

	tf.flags.DEFINE_integer("rnn_hidden", 512, "Number of hidden states for LSTM (default: 128)")
	tf.flags.DEFINE_string('output_dir', None, 'output directory (default: results/ )')
	# model parameters 
	tf.flags.DEFINE_string('filter_sizes', '3,4,5', 'Comma-separated filter sizes (default: 2,3,4,5)')
	tf.flags.DEFINE_integer('num_filters', 32, 'Number of filters per filter size (default 4)')
	tf.flags.DEFINE_integer('fc_hidden_layers', 1, 'Number of fc hidden layers (default 1)')
	tf.flags.DEFINE_integer('fc_hidden_nodes', 128, 'Number of fc hidden layers (default 128)')
	tf.flags.DEFINE_float('dropout_keep_prob', 1.0, 'Dropout keep probability (default: 1.0)')
	tf.flags.DEFINE_float('l2_reg_lambda', 1e-3, 'L2 regularization lambda (default: 1e-3)')
	tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate while training (default: 1e-3)')
	tf.flags.DEFINE_integer('random_seed', 13, 'Random seeds for reproducibility (default: 13)')


	# Training parameters.  ??? should decide on this latter
	tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 128)")
	tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
	tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
	tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
	# Misc Parameters
	tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
	tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
			print("{}={}".format(attr.upper(), value))
	return FLAGS


if __name__ == '__main__':

	#########################################
	#   Load arguments and configure file   #
	#########################################
	# Process CML arguments
	FLAGS = handle_flags()

	random.seed(FLAGS.random_seed)
	np.random.seed(FLAGS.random_seed)
	tf.set_random_seed(FLAGS.random_seed)

	file_path = "../../data/"
	train_index_list=list(map(int, FLAGS.train_data_set.split(',')))
	test_index_list = list(map(int,FLAGS.test_data_set.split(',')))
	fea_train,lbl_train,fea_valid,lbl_valid,fea_test,lbl_test = data_helpers.load_dataset(file_path,train_index_list,test_index_list,FLAGS.start_time,FLAGS.sequence_length)
	print("finish loading data set...")
	print("fea_train "+str(fea_train.shape))
	print("lbl_train "+str(lbl_train.shape))
	print("fea_valid "+str(fea_valid.shape))
	print("lbl_valid "+str(lbl_valid.shape))
	print("fea_test "+str(fea_test.shape))
	print("lbl_test "+str(lbl_test.shape))


	with tf.Graph().as_default():
		# set up session configuration
		session_conf = tf.ConfigProto(
		  allow_soft_placement=FLAGS.allow_soft_placement,
		  log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		# get into training session
		with sess.as_default():
			# construct a model with parameters 
			model = RNN(sequence_length = FLAGS.sequence_length,
				num_classes = 4, 
				num_electrode = 22,
				rnn_hidden = FLAGS.rnn_hidden,
				filter_sizes = list(map(int, FLAGS.filter_sizes.split(','))),
				num_filters = FLAGS.num_filters,
				fc_hidden_layers = FLAGS.fc_hidden_layers,
				l2_reg_lambda=FLAGS.l2_reg_lambda,
				fc_hidden_nodes = FLAGS.fc_hidden_nodes)


			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
			grads_and_vars = optimizer.compute_gradients(model.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


			# Output directory for models and summaries
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.pardir, "runs", timestamp))
			print("Writing to {}\n".format(out_dir))

			# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

			# Initialize all variables
			print('Global var initializer')
			sess.run(tf.global_variables_initializer())


			def train_step(batch_x,batch_y):
				feed_dict = {
						model.input_x:batch_x,
						model.input_y: batch_y,
						model.dropout_keep_prob: FLAGS.dropout_keep_prob
					}   
				_, step, loss = sess.run(
						[train_op, global_step, model.loss],
						feed_dict)
				time_str = datetime.datetime.now().isoformat()
				print("{}: step {}, loss {:g}".format(time_str, step, loss))
				return loss


			def evaluation(y1, y2,description):
				y_answer = y1.flatten()
				y_pred = np.argmax(y2,axis = 1)
				acc = accuracy_score(y_answer, y_pred)
				'''
				print("acc %4f"%acc)
				method_list = ['macro']
				print("method    precision   recall   f1_score")
				for method in method_list:
					precision = precision_score(y_answer,y_pred,average = method)
					recall = recall_score(y_answer,y_pred,average = method)
					f1 = f1_score(y_answer,y_pred,average = method)
					print("%s    %4f    %4f    %4f" % (method, precision, recall, f1))
				print("confusion_matrix")
				print(confusion_matrix(y_answer,y_pred))
				'''
				return acc



			def dev_step(batch_x, batch_y, description):
				feed_dict = {
						model.input_x:batch_x,
						model.input_y:batch_y,
						model.dropout_keep_prob: 1.0  #inverted dropout impl
					}   

				step, y_pred = sess.run([global_step, model.pred],feed_dict)
				time_str = datetime.datetime.now().isoformat()
				return evaluation(batch_y, y_pred,description)


			batches = data_helpers.batch_iter(            
					list(zip(fea_train, lbl_train)),
					FLAGS.batch_size, 
					FLAGS.num_epochs)

			# Batch training 
			train_acc_list = []
			valid_acc_list = []
			for batch in batches:
				batch_x,batch_y = zip(*batch) # x_batch, y_batch = zip(*batch)
				loss = train_step(batch_x, batch_y)
				#train_loss_list.append(loss)
				current_step = tf.train.global_step(sess, global_step)
				if current_step % FLAGS.checkpoint_every == 0:
					train_acc = dev_step(fea_train,lbl_train,"train")
					#train_acc_list.append(train_acc)
					#dev_step(body_valid,fea_valid,lbl_valid,"validation")
					valid_acc = dev_step(fea_valid,lbl_valid,"valid")
					test_acc = dev_step(fea_test,lbl_test,"test")
					print("train acc: %4f   valid acc: %4f     test acc:%4f" % (train_acc,valid_acc,test_acc))
					train_acc_list.append(train_acc)
					valid_acc_list.append(valid_acc)


			print("train_acc_list "+str(train_acc_list))
			print("valid_acc_list "+str(valid_acc_list))
































