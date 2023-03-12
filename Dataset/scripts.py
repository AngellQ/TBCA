import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset




class SampleReader:

    def one_hot(seq):

        base_map = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'G': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]}

        code = np.empty(shape=(len(seq), 4))
        for location, base in enumerate(seq, start=0):
            code[location] = base_map[base]

        return code


    def __init__(self, file_name):
  
        self.seq_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/' + file_name + '/Sequence/'
        self.shape_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/' + file_name + '/Shape/'

    def get_seq(self, Test=False):

        if Test is False:
            row_seq = pd.read_csv(self.seq_path + 'Train_seq.csv', sep=' ', header=None)
        else:
            row_seq = pd.read_csv(self.seq_path + 'Test_seq.csv', sep=' ', header=None)

        seq_num = row_seq.shape[0]
        seq_len = len(row_seq.loc[0, 1])

        completed_seqs = np.empty(shape=(seq_num, seq_len, 4))
        completed_labels = np.empty(shape=(seq_num, 1))
        for i in range(seq_num):   
            completed_seqs[i] = one_hot(row_seq.loc[i, 1])
            # completed_seqs[i] = sequence2num(row_seq.loc[i, 1])
            completed_labels[i] = row_seq.loc[i, 2]
        completed_seqs = np.transpose(completed_seqs, [0, 2, 1])

        return completed_seqs, completed_labels

    def get_shape(self, shapes, Test=False):

        shape_series = []

        if Test is False:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Train' + '_' + shape + '.csv'))
        else:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Test' + '_' + shape + '.csv'))

        """
            seq_num = shape_series[0].shape[0]
            seq_len = shape_series[0].shape[1]
        """
        completed_shape = np.empty(shape=(shape_series[0].shape[0], len(shapes), shape_series[0].shape[1]))
        #print(len(shapes))

        for i in range(len(shapes)):
            shape_samples = shape_series[i]
            #print(shape_samples.loc[0])
            #loc函数主要通过行标签索引行数据，
            for m in range(shape_samples.shape[0]):
                completed_shape[m][i] = shape_samples.loc[m]
        completed_shape = np.nan_to_num(completed_shape)

        return completed_shape


class SSDataset_690(Dataset):

    def __init__(self, file_name, Test=False):
        shapes = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']

        sample_reader = SampleReader(file_name=file_name)

        self.completed_seqs, self.completed_labels = sample_reader.get_seq(Test=Test)
        self.completed_shape = sample_reader.get_shape(shapes=shapes, Test=Test)

    def __getitem__(self, item):
        return self.completed_seqs[item], self.completed_shape[item], self.completed_labels[item]

    def __len__(self):
        return self.completed_seqs.shape[0]