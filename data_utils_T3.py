import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random
import logging

logging.basicConfig(filename='errors.log', level=logging.DEBUG)

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

def genSpoof_list( dir_meta,is_train=False,is_eval=False, is_dev=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
    

    if (is_train):
        for line in l_meta:
             key, subset, label = line.strip().split()
             if subset == 'train':
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    
    if (is_dev):
        for line in l_meta:
             key, subset, label = line.strip().split()
             if subset == 'dev':
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    
    elif(is_eval):
        for line in l_meta:
            key, subset, label = line.strip().split()
            file_list.append(key)
        return file_list



def pad(x, utt_id, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    if x_len < 1:
        logging.error("file {} has 0 size".format(utt_id))
        exit(0)

    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_for(Dataset):
	def __init__(self,args,list_IDs, labels, base_dir, algo):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.algo=algo
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            # X,fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000) 
            X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
            X_pad= pad(X,utt_id,self.cut)
            x_inp= Tensor(X_pad)
            target = self.labels[utt_id]
            
            return x_inp, target
            
class Dataset_for_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
            X_pad = pad(X,utt_id,self.cut)
            
            x_inp = Tensor(X_pad)
            return x_inp, utt_id

