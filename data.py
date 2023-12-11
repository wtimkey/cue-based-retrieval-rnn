import torch
import h5py
import random
import numpy as np

class Dictionary(object):
    """ Maps between observations and indices """
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.idx2word = [x[:-1] for x in f.readlines()][:-1]
        self.word2idx = {tok:i for i, tok in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)
    
#dataset is a train chunk, valid, or test data in tensorized form with
#aggregate attention masks, loss masks, inputs, etc. as fields
class Dataset(object):
    def __init__(self, datafile, device, chunk_idxs, batch_size=512, chunk_size=512, train=False, max_input_len=128):
        if(datafile is None):
            return None
        self.input_masks = None
        self.input_toks = None
        self.aux_labels = None
        self.lm_loss_pad_masks = None
        self.type_loss_masks = None
        self.num_terminals = None
        if(train):
            reshuffle_train_idxs = [random.sample(list(range(chunk_size)), chunk_size) for i in range(len(chunk_idxs))]
        for k in datafile.keys():
            if(train):
                curr_data = np.swapaxes(np.stack([datafile[k][x:x+chunk_size][reshuffle_train_idxs[i]] for i, x in enumerate(chunk_idxs)]), 0, 1)
            else:
                curr_data = np.stack([datafile[k][x:x+batch_size] for x in chunk_idxs])
                
            if(k == 'input_masks'):
                self.input_masks = torch.from_numpy(np.log(curr_data))
            if(k == 'input_toks'):
                self.input_toks = torch.from_numpy(np.swapaxes(curr_data, 1, 2))
            if(k == 'aux_labels'):
                self.aux_labels = torch.from_numpy(np.swapaxes(curr_data, 1, 2))
            if(k == 'length'):
                self.lm_loss_pad_masks = self.populate_lm_pad_mask_array(curr_data, len(datafile['input_toks'][0]))
            if(k == 'ent_loss_masks'):
                self.type_loss_masks = torch.from_numpy(np.swapaxes(curr_data, 1, 2))
                self.num_terminals = torch.from_numpy(curr_data.sum(axis=1))
    
    def __len__(self):
        return len(self.input_toks)

    #generate mask over loss on padding tokens, given list of example lengths
    def populate_lm_pad_mask_array(self, lens, max_input_len):
        init_array = np.zeros((lens.shape[0], lens.shape[1], max_input_len), dtype=bool)
        for i in range(init_array.shape[0]):
            for j in range(init_array.shape[1]):
                init_array[i,j, :lens[i,j]] = True
        return torch.from_numpy(np.swapaxes(init_array, 1, 2))
    

class SentenceCorpus(object):
    """ Loads train/dev/test corpora and dictionary """
    def reset_shuffled_ex_order(self):
        self.shuffled_ex_order = list(range(0, self.num_train_exs, self.h5_access_chunk_size)[:-1])
        random.shuffle(self.shuffled_ex_order)
    
    def __init__(self, path, vocab_file, device, model_type='CBRRNN',
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt',
                 batch_size=512,
                 h5_access_chunk_size=512,#number of batches per chunk
                 test_flag=False,
                 seed=1111,
                 aux_objective_flag=False, aux_vocab_file='aux_labels.txt'):
        
        random.seed(seed)
        self.device = device
        self.aux_objective = aux_objective_flag
        self.batch_size = batch_size
        self.dictionary = Dictionary(path + vocab_file)
        self.h5_access_chunk_size = h5_access_chunk_size

        if(model_type=='CBRRNN' or model_type=='LSTM'):
            self.input_label = 'input_toks'
        else:
            self.input_label = 'actions'

        if(self.aux_objective):
            self.aux_dictionary = Dictionary(aux_vocab_file)
        else:
            self.aux_dictionary = None
        if(test_flag):
            self.testfile = h5py.File(testfname, "r")
            self.num_test_exs = len(self.testfile[self.input_label])
            self.trainfile = None
            self.num_test_exs = 0
            self.validfile = None
            self.num_valid_exs = 0
        else:
            self.testfile = None
            self.trainfile = h5py.File(path + trainfname, "r")
            self.validfile = h5py.File(path + validfname, "r")
            self.num_train_exs = len(self.trainfile[self.input_label])
            self.num_valid_exs = len(self.validfile[self.input_label])
            self.num_test_exs = 0
            self.reset_shuffled_ex_order()

        self.valid = Dataset(self.validfile, self.device, chunk_idxs=list(range(0, self.num_valid_exs, self.batch_size))[:-1], batch_size=self.batch_size)
        self.test = Dataset(self.testfile, self.device, chunk_idxs=list(range(0, self.num_test_exs, self.batch_size))[:-1], batch_size=self.batch_size)
        #load the next chunk of training data into memory, organized into batches
        #todo: deal with case when out of data!
    def get_train_chunk(self):
        curr_chunk_batches = self.shuffled_ex_order[:self.batch_size]
        if(len(curr_chunk_batches) == 0): #out of data.
            self.reset_shuffled_ex_order()
            return None
        self.shuffled_ex_order = self.shuffled_ex_order[self.batch_size:]
        return Dataset(self.trainfile, self.device, curr_chunk_batches, train=True)
