"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import time
import tensorflow as tf
import h5py
import scipy
from scipy import signal
import os
from sklearn.model_selection import train_test_split
from numpy import newaxis
import braindecode.datautil.signalproc as signalproc

def stft2phi(xs,seg=50,reshape=False): 
    '''
    using seg=50(250Hz) to select a window of 0.2s
    generate picture dim: ()
    '''
    _,_, Zxx = signal.stft(xs,nperseg=seg)
    # reshape size
    Zxx_r = np.sqrt(Zxx.real**2+Zxx.imag**2)
    Zxx_phi = np.angle(Zxx)
    return (Zxx_r,Zxx_phi)

def istftfromphi(Zxx, seg=50):
    (Zxx_r, Zxx_phi) = Zxx
    Zxx_rec_real = Zxx_r * np.cos(Zxx_phi)
    Zxx_rec_imag = Zxx_r * np.sin(Zxx_phi)
    Zxx_rec = Zxx_rec_real+Zxx_rec_imag*1j
    _, x_rec = signal.istft(Zxx_rec,nperseg=seg)
    x_rec = np.asarray(x_rec, dtype=np.float64)
    return x_rec

class Data(object):
    '''
    This class is 
    '''
    def __init__(self):
        self.datadict = {}
        self.dataframe = []
        self.datalabel = []
        self.gen_frame = None
        self.gen_spec = None
        self.spectrum = None
        self.timeDim = None
        self.freDim = None
        self.channel = 2
        self.electrode_num = 22
        self.subject_num = 9
        self.generate_size = 100
        self.timeStamp = 1000
        self.data_dir = './dataset/EEG_gen_dump.pickle'
        self.train = None
        self.test = None
        self.timeWindow = 1000

    def train_test_split(self,mode,newdim=True,timesize=1000):
        #X_train, X_test, y_train, y_test = train_test_split(self.dataframe, self.datalabel, test_size=0.2)
        if mode['train_obj'] > 0:
            X = self.dataframe[mode['train_obj']-1,:]
            y = np.asarray(self.datalabel[mode['train_obj']-1],dtype=np.int32) - 769
        else:
            X = self.dataframe.reshape((-1,self.electrode_num, self.timeStamp))
            y = np.asarray(self.datalabel.reshape((-1)),dtype=np.int32) - 769

        if timesize != 1000:
            X = X[:,:,0:timesize]
            self.timeWindow = timesize

        if mode['test_size'] > 0 and mode['train_size'] > 0:
            test_mask = np.random.choice(X.shape[0], mode['test_size'], replace=False)
            train_mask = np.random.choice(X.shape[0], mode['train_size'], replace=False)
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask],y[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if newdim:
            X_train = X_train[..., newaxis]
            X_test = X_test[..., newaxis]

        self.train = (X_train, y_train)
        self.test = (X_test,y_test)

    def load_data(self, filename=None, params=None,create_frame_seg=50):
        self.dataframe = np.zeros((9,288,22,1000))
        self.datalabel = np.zeros((9,288))
        for i in range(1,10,1):
            AT_slice = h5py.File('dataset/A0'+str(i)+'T_slice.mat', 'r')
            X = np.copy(AT_slice['image'])
            X = X[:,:22,:] # select first 22 channels
            y = np.copy(AT_slice['type'])
            y = y[0,0:X.shape[0]:1]
            y = np.asarray(y, dtype=np.int32)
            # replace NaN as 0
            X[np.isnan(X)] = 0
            self.datadict['A0'+str(i)+'T'] = (X,y)
            self.dataframe[i-1,:,:,:] = X
            self.datalabel[i-1,:] = y
        # preprocessing using braincode
        for i in range(9):
            for j in range(228):
                self.dataframe[i,j] = np.transpose(signalproc.highpass_cnt(np.transpose(self.dataframe[i,j]), 4, 1000, filt_order=3, axis=0))
                self.dataframe[i,j] = np.transpose(signalproc.exponential_running_standardize(np.transpose(self.dataframe[i,j]), factor_new=0.001, init_block_size=None, eps=0.0001))
        print("Data filtered")
        if create_frame_seg:
            self.create_frame(create_frame_seg)
        print("Data fully loaded!")

    def create_frame(self, seg=50):
        (spec_r, spec_phi) = stft2phi(self.dataframe,seg)
        self.timeDim = spec_r.shape[-1]
        self.freDim = spec_r.shape[-2]
        self.spectrum = np.zeros((9,288,22,self.freDim,self.timeDim,self.channel))
        self.spectrum[:,:,:,:,:,0] = spec_r
        self.spectrum[:,:,:,:,:,1] = spec_phi
        self.load(self.data_dir)
        #self.save(self.data_dir)

    def get_slice(self, electrode, subject=None, target=None):
        '''get full data matrix by electrode(necessary), subject and target(optional)'''
        if subject > 0:
            X = self.spectrum[subject,:,electrode,:,:,:]
            label = np.asarray(self.datalabel[subject,:].reshape((-1)),dtype=np.int32)
        else:
            print('[Warning]No label or Invalid subject label! Return data in all labels...')
            X = self.spectrum[:,:,electrode,:,:,:]
            label = np.asarray(self.datalabel.reshape((-1)),dtype=np.int32)

        X = X.reshape((-1,self.freDim,self.timeDim,self.channel))
        y = np.zeros((len(label),4))
        y[range(len(label)),label-769] = 1
        if target in [769,770,771,772]:
            # select target data
            mask = np.where(label==target)[0].reshape((-1))
            X = X[mask,:,:,:]
            y = y[mask,:]
        else:
            print("[Warning]No label or Invalid target label! Return data in all labels...")
        return X, y

    def load_gen(self,samples,sample_label,gen_id,params=None):
        self.gen_spec[gen_id,:] = samples
        self.gen_label[gen_id,:] = sample_label
        self.gen_frame[gen_id,:] = istftfromphi((self.gen_spec[gen_id,:,:,:,:,0],self.gen_spec[gen_id,:,:,:,:,0]))
        self.save(self.data_dir)

    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump((self.gen_spec, self.gen_label, self.gen_frame) , f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)

    def load(self, filename):
        if os.path.isfile(filename):
            f = open(filename,'rb')
            self.gen_spec, self.gen_label, self.gen_frame = pickle.load(f)
            print("Loaded data object from", filename)
        else:
            self.gen_spec = np.zeros((3,self.generate_size,22,self.freDim,self.timeDim,self.channel))
            self.gen_frame = np.zeros((3,self.generate_size,22,1000))
            self.gen_label = np.zeros((3,self.generate_size))

if __name__ == "__main__":
    ### data processing debug
    data = Data()
