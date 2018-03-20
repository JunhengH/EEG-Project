##!/usr/bin/env python3
import sys, os
import numpy as np
import h5py
try:
	import ujson as json
except:
	print('Cannot import ujson, import json instead.', file=sys.stderr)
	import json

try:
	from smart_open import smart_open as open
except:
	print('smart_open inexists, use original one instead.', file=sys.stderr)
	
import tensorflow as tf

def filter_data(X,y):
	#filter out NAN data
	filter_index = []
	for index in range(X.shape[0]):
		data = X[index,:,:]
		if np.isfinite(data).all() == False or (np.isnan(data).any() == True):
			filter_index.append(index)
	print("filter index: "+str(filter_index))
	X = np.delete(X,filter_index,axis = 0)
	y = np.delete(y,filter_index,axis = 0)
	return X,y



#we get train,valid,test from the test index
def read_in_test(file_path,test_index_list,validation_num):
	train_fea,valid_fea,test_fea,train_lbl,valid_lbl,test_lbl = None,None,None,None,None,None
	first = True

	for index in test_index_list:
		tmp_data = h5py.File(file_path+'A0'+str(index)+'T_slice.mat', 'r')
		print("finish reading file "+str(index))
		X = np.copy(tmp_data['image'])
		y = np.asarray(np.copy(tmp_data['type'])[0,0:X.shape[0]:1],dtype=np.int32)-769
		y = np.array(y).reshape((len(y), 1))
		print("file "+str(index)+" X origin shape"+str(X.shape))
		print("file "+str(index)+" y origin shape"+str(y.shape))
		X,y = filter_data(X,y)
		print("file "+str(index)+" X valid shape"+str(X.shape))
		print("file "+str(index)+" y valid shape"+str(y.shape))
		#split data into train,valid,test
		new_index = np.arange(X.shape[0])
		np.random.shuffle(new_index)
		X,y = X[new_index],y[new_index]
		tmp_train_fea,tmp_valid_fea,tmp_test_fea = X[50+validation_num:,:,:],X[50:50+validation_num,:,:],X[:50,:,:]
		tmp_train_lbl,tmp_valid_lbl,tmp_test_lbl = y[50+validation_num:,:],y[50:50+validation_num,:],y[:50,:]
		train_fea = tmp_train_fea if first else np.concatenate((train_fea,tmp_train_fea),axis = 0)
		valid_fea = tmp_valid_fea if first else np.concatenate((valid_fea,tmp_valid_fea),axis = 0)
		test_fea = tmp_test_fea if first else np.concatenate((test_fea,tmp_test_fea),axis = 0)
		train_lbl = tmp_train_lbl if first else np.concatenate((train_lbl,tmp_train_lbl),axis = 0)
		valid_lbl = tmp_valid_lbl if first else np.concatenate((valid_lbl,tmp_valid_lbl),axis = 0)
		test_lbl = tmp_test_lbl if first else np.concatenate((test_lbl,tmp_test_lbl),axis = 0)
		first = False

	return train_fea,valid_fea,test_fea,train_lbl,valid_lbl,test_lbl


#we get the rest of training set from the not test index
def read_in_rest_train(file_path,rest_train_index):
	train_fea,train_lbl = None,None
	first = True
	for index in rest_train_index:
		tmp_data = h5py.File(file_path+'A0'+str(index)+'T_slice.mat', 'r')
		print("finish reading file "+str(index))
		X = np.copy(tmp_data['image'])
		y = np.asarray(np.copy(tmp_data['type'])[0,0:X.shape[0]:1],dtype=np.int32)-769
		y = np.array(y).reshape((len(y), 1))
		X,y = filter_data(X,y)
		print("file "+str(index)+" X valid shape"+str(X.shape))
		print("file "+str(index)+" y valid shape"+str(y.shape))
		new_index = np.arange(X.shape[0])
		np.random.shuffle(new_index)
		X,y = X[new_index],y[new_index]
		if first != True:
			print("train_fea "+str(train_fea.shape))
			print("X "+str(X.shape))
		train_fea = X if first else np.concatenate((train_fea,X),axis = 0)
		train_lbl = y if first else np.concatenate((train_lbl,y),axis = 0)
		first = False
	return train_fea,train_lbl



def preprocess(train_fea,valid_fea,test_fea):
	#normalize each channel to [-1,1], filter the extreme cases
	for channel in range(train_fea.shape[1]):
		data = train_fea[:,channel,:].flatten()
		#quantile_list = np.array([0,0.001,0.005,.01,.1,.2,.3,.5,.7,.8,.9,.93,.95,.98,.99,.995,.999,1])
		#stat = np.percentile(data,quantile_list*100)
		lower_bound,upper_bound = np.percentile(data,[0.1,99.9])
		train_fea[:,channel,:] = np.minimum(1,np.maximum(0,(train_fea[:,channel,:]-lower_bound)/(upper_bound-lower_bound)))
		valid_fea[:,channel,:] = np.minimum(1,np.maximum(0,(valid_fea[:,channel,:]-lower_bound)/(upper_bound-lower_bound)))
		test_fea[:,channel,:] = np.minimum(1,np.maximum(0,(test_fea[:,channel,:]-lower_bound)/(upper_bound-lower_bound)))

	#print(train_fea[0,:,:20])
	#print("****************")
	#print(test_fea[0,:,:20])

	return train_fea,valid_fea,test_fea


def load_dataset(file_path,train_index_list,test_index_list,start_time,sequence_length):
	if train_index_list[0] == -1:
		train_index_list = np.arange(9)+1
	if test_index_list[0] == -1:
		test_index_list = np.arange(9)+1

	if len(test_index_list) == 1 and len(train_index_list) > 1:
		validation_num = 50
	else:
		validation_num = 36

	train_fea,valid_fea,test_fea,train_lbl,valid_lbl,test_lbl = read_in_test(file_path,test_index_list,validation_num)
	print("read in rest train....")
	rest_train_index = np.setdiff1d(train_index_list,test_index_list)
	if len(rest_train_index) > 0:
		rest_train_fea,rest_train_lbl = read_in_rest_train(file_path,rest_train_index)
		train_fea = np.concatenate((train_fea,rest_train_fea),axis = 0)
		train_lbl = np.concatenate((train_lbl,rest_train_lbl),axis = 0)

	#filter and preprocess the data
	#preprocess(train_fea,valid_fea,test_fea)
	end_time = start_time + sequence_length
	return train_fea[:,:22,start_time:end_time],train_lbl,valid_fea[:,:22,start_time:end_time],valid_lbl,test_fea[:,:22,start_time:end_time],test_lbl


def batch_iter(data_, batch_size, num_epochs, shuffle=True):
	data = np.array(data_)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]  #return a generator and the item are batches






