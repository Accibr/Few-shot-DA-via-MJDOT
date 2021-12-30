import os
import logging
import numpy as np
import torch
import pandas as pd

join = os.path.join
logger = logging.getLogger()


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class XianDataset(dotdict):

    def __init__(self, IMG_DIR, IMG_NAME, feature_norm='none', verbose=True):#IMG_DIR_SRC, IMG_DIR_TAR
        super().__init__()
        assert feature_norm == 'none' or \
               feature_norm == 'l2' or \
               feature_norm == 'unit-gaussian'

        # feature_s = pd.read_csv(IMG_DIR_SRC + '_resnet50_feats.csv', header=None)
        # label_s = pd.read_csv(IMG_DIR_SRC + '_resnet50_labels.csv', header=None)
        # feature_t = pd.read_csv(IMG_DIR_TAR + '_resnet50_feats.csv', header=None)
        # label_t = pd.read_csv(IMG_DIR_TAR + '_resnet50_labels.csv', header=None)

        # self.feature_s = feature_s.values.astype('float32')
        # self.label_s = np.int64(label_s.values)  # .ravel()
        # self.feature_t = feature_t.values.astype('float32')
        # self.label_t = np.int64(label_t.values)  # .ravel()

#officehome+office31
        tmpS = np.loadtxt(IMG_DIR + IMG_NAME + '/source_train.csv', dtype=np.str, delimiter=",")
        tmpT = np.loadtxt(IMG_DIR + IMG_NAME + '/target_train.csv', dtype=np.str, delimiter=",")
        # tmpS = np.loadtxt('DATAMISS/Office-Home_20_3/' + IMG_DIR + '/source_train.csv', dtype=np.str, delimiter=",")
        # tmpT = np.loadtxt('DATAMISS/Office-Home_20_3/' + IMG_DIR + '/target_train.csv', dtype=np.str, delimiter=",")
        self.feature_s = tmpS[0:, 0:2048].astype('float32')
        self.label_s = tmpS[0:, 2048].astype(np.float)
        self.feature_t = tmpT[0:, 0:2048].astype('float32')
        self.label_t = tmpT[0:, 2048].astype(np.float)


        # if self.feature_s.shape[0] < self.feature_s.shape[1]: self.feature_s = self.feature_s.T # if transposed
        # if self.feature_t.shape[0] < self.feature_t.shape[1]: self.feature_t = self.feature_t.T  # if transposed
        self.label_s.min()  # labels start from 0
        self.label_t = self.label_t - self.label_t.min()

        if feature_norm == 'l2':
            self.feature_s /= np.linalg.norm(self.feature_s, axis=1, keepdims=True)
            self.feature_t /= np.linalg.norm(self.feature_t, axis=1, keepdims=True)

        self.d_ft = self.feature_s.shape[1]
        self.Call = np.unique(self.label_s)
        self.n_Call = self.Call.shape[0]



def index_labels(labels, classes, check=True):
    """
    Indexes labels in classes.

    Arg:
        labels:  [batch_size]
        classes: [n_class]
    """
    indexed_labels = np.searchsorted(classes, labels)
    if check:
        assert np.all(np.equal(classes[indexed_labels], labels))

    return indexed_labels
