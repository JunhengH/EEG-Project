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
        self.data = []
        self.dataframe = []
        self.datalabel = []
        self.train_data = []
        self.test_data = []
        self.valid_data = []
        self.gen_frame = None
        self.gen_data = None
        self.spec_r = None
        self.spec_phi = None
        self.timeDim = 0
        self.freDim = 0

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
        if create_frame_seg:
            self.create_frame(create_frame_seg)
        print("Data fully loaded!")

    def create_frame(self, seg=50):
        (self.spec_r, self.spec_phi) = stft2phi(self.dataframe,seg)
        self.timeDim = self.spec_r.shape[-1]
        self.freDim = self.spec_r.shape[-2]


    def get_slice(self, electrode, subject=None, target=None):
        '''get full data matrix by electrode(necessary), subject and target(optional)'''
        if subject > 0:
            X_r = self.spec_r[subject,:,electrode,:,:]
            X_phi = self.spec_phi[subject,:,electrode,:,:]
            label = np.asarray(self.datalabel[subject,:].reshape((-1)),dtype=np.int32)
        else:
            print('[Warning]No label or Invalid subject label! Return data in all labels...')
            X_r = self.spec_r[:,:,electrode,:,:]
            X_phi = self.spec_phi[:,:,electrode,:,:]
            label = np.asarray(self.datalabel.reshape((-1)),dtype=np.int32)

        X_r = X_r.reshape((-1,self.freDim,self.timeDim))
        X_phi = X_phi.reshape((-1,self.freDim,self.timeDim))
        y = np.zeros((len(label),4))
        y[range(len(label)),label-769] = 1
        if target in [769,770,771,772]:
            # select target data
            mask = np.where(label==target)[0].reshape((-1))
            X_r = X_r[mask,:,:]
            X_phi = X_phi[mask,:,:]
            y = y[mask,:]
        else:
            print("[Warning]No label or Invalid target label! Return data in all labels...")
        return X_r, X_phi, y

    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)

if __name__ == "__main__":
    ### data processing debug
    data = Data()
