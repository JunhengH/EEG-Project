import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np



class RNN(object):
	"""
	A CNN for activity classification.
	Uses a convolutional layer, followed by max-pooling and softmax layer.
	"""
	def __init__(self, 
		sequence_length, 
		num_classes, 
		num_electrode = 22,
		rnn_hidden = 512,
		filter_sizes = [3, 5, 7],
		num_filters = 32,
		fc_hidden_layers = 1,
		l2_reg_lambda=0.0,
		fc_hidden_nodes = 128):

		self.rnn_hidden = rnn_hidden
		self.sequence_length = sequence_length
		self.num_classes = num_classes
		self.num_electrode = num_electrode
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.fc_hidden_layers = fc_hidden_layers
		self.l2_reg_lambda = l2_reg_lambda
		self.fc_hidden_nodes = fc_hidden_nodes

		self.build_model()




	def build_model(self):
		self.input_x = tf.placeholder(tf.float32, [None,self.num_electrode,self.sequence_length], name="input_x")
		print ("input_x.shape: "+str(self.input_x.get_shape()))
		self.input_y = tf.placeholder(tf.int32, [None, 1], name="input_y")
		print ("input_y.shape: "+str(self.input_y.get_shape()))
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.unstacked_x = tf.unstack(self.input_x, self.sequence_length, axis=2)
		print("unstack_x shape:"+str(len(self.unstacked_x)))
		print("element 0 shape:"+str(self.unstacked_x[0].shape))

		# Keeping track of l2 regularization loss (optional)
		self.l2_loss = tf.constant(0.0)

		self.build_lstm_layer()
		self.build_conv_layer()
		self.build_hidden_layers()

		#calculate the loss
		reformat_y = tf.reshape(self.input_y,[-1])
		self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(reformat_y, self.num_classes), logits = self.pred))\
		+self.l2_reg_lambda * self.l2_loss
		print(self.loss.shape)



	def build_lstm_layer(self):
		#lstm layer
		with tf.device('/gpu:0'),tf.name_scope("lstm") :
			lstm_cell = rnn.BasicLSTMCell(self.rnn_hidden)
			#lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
			outputs, states = rnn.static_rnn(lstm_cell, self.unstacked_x, dtype=tf.float32)
			print("outputs.get_shape() "+str(len(outputs)))
			print("states.get_shape() "+str(len(states)))
			self.lstm_output = outputs[-1]
			print("lstm_output.size() "+str(self.lstm_output.get_shape()))


	def build_conv_layer(self):
		self.expand_x = tf.expand_dims(self.input_x, -1)
		pooled_outputs = []
		for i, filter_size in enumerate(self.filter_sizes):
			with tf.device('/gpu:0'), tf.name_scope('conv-maxpool-%s' % filter_size):
				#convolutional layer
				filter_shape = [self.num_electrode,filter_size, 1, self.num_filters] 
				#[filter_height,filter_width,in_channel,out_channel]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.expand_x,
					W,
					strides=[1, 1, 1, 1], #should I expand dims here?
					padding="VALID",
					name="conv")
				print("conv shape: "+str(conv.get_shape()))

				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 

				pooled = tf.nn.max_pool(
					h,
					ksize=[1,1,self.sequence_length - filter_size + 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				print('pooled shape:'+str(pooled.get_shape()))
				pooled_outputs.append(pooled)

		num_filters_total = self.num_filters * len(self.filter_sizes)
		self.cnn_output = tf.concat(pooled_outputs, 3)
		print("cnn_output shape1:"+str(self.cnn_output.get_shape()))
		self.cnn_output = tf.reshape(self.cnn_output,[-1,num_filters_total])
		print("cnn output shape2: "+str(self.cnn_output.get_shape()))



	def build_hidden_layers(self):
		self.final_features = tf.concat([self.lstm_output,self.cnn_output],axis = 1)
		#self.final_features = self.cnn_output
		print("final feature shape"+str(self.final_features.get_shape()))

		last_output = self.final_features
		last_size = int(self.final_features.get_shape()[-1])
		self.hidden_layer_output = []

		for i in range(self.fc_hidden_layers):
			with tf.name_scope('hidden-layer-%d' % i):
				W = tf.Variable(tf.truncated_normal([last_size, self.fc_hidden_nodes], stddev=0.1), name='W')
				b = tf.Variable(tf.constant(0.1, shape=[self.fc_hidden_nodes]), name='b')
				self.l2_loss += tf.nn.l2_loss(W)
				self.l2_loss += tf.nn.l2_loss(b)
				output = tf.matmul(self.hidden_layer_output[-1] if i > 0 else self.final_features, W) + b
				output = tf.nn.tanh(output)
				output = tf.nn.dropout(output, self.dropout_keep_prob)
				print('layer #%d' % i)
				self.hidden_layer_output.append(output)
				last_size = self.fc_hidden_nodes
				last_output = output

		with tf.name_scope('final'):
			W = tf.Variable(tf.truncated_normal([int(last_output.get_shape()[-1]) ,self.num_classes], stddev=0.1), name='W')
			b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')

			self.l2_loss += tf.nn.l2_loss(W)
			self.l2_loss += tf.nn.l2_loss(b)

			self.pred = tf.add(tf.matmul(self.hidden_layer_output[-1], W),  b, name='pred')
			print("self.pred shape "+str(self.pred.get_shape()))









if __name__ == '__main__':
	har = RNN(
			sequence_length = 1000, 
			num_electrode = 22,
			rnn_hidden = 128,
			num_classes = 4,
			l2_reg_lambda = 0.02)