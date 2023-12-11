import pickle
from tqdm import tqdm
from collections import Counter
import random
import h5py
import argparse
random.seed(1111)

parser = argparse.ArgumentParser(description='Generate training/valid/test datafiles for training CBRRNN language models')

parser.add_argument('--train_input_fname', type=str, default=None,
                    help='name of input training data file')
parser.add_argument('--valid_input_fname', type=str, default=None,
                    help='name of input validation data file')
parser.add_argument('--test_input_fname', type=str, default=None,
                    help='name of input test data file')
parser.add_argument('--seq_len', type=int, default=64,
                    help='maximum sequence length at training time')
parser.add_argument('--vocab_size', type=int, default=2**16)
parser.add_argument('--lower', action='store_true',
                    help='lowercase all words in data')
parser.add_argument('--ccg_supertags', action='store_true',
                    help='compute vocabulary for CCG supertags')
parser.add_argument('--aux_vocab_size',  type=int, default=None,
                    help='compute vocabulary for CCG supertags')
parser.add_argument('--vocab_fname', type=str, default='vocab_f.txt',
                    help='default location to store vocab file')
parser.add_argument('--aux_vocab_fname', type=str, default='aux_vocab_f.txt',
                    help='default location to store vocab file')
args = parser.parse_args()

def lower_num(word):
    word = word.replace("-LRB-", '(')
    word = word.replace("-LCB-", '{')
    word = word.replace("-LSB-", '[')
    word = word.replace("-RRB-", ')')
    word = word.replace("-RCB-", '}')
    word = word.replace("-RSB-", ']')
    if(args.lower):
        word = word.lower()

    word_num = word.replace('.', '')
    word_num = word_num.replace(',', '')
    word_num = word_num.replace('-', '')
    if(word_num.isnumeric()):
        return '<num>'
    else:
        return word
    
def unk(word, curr_vocab):
    if(word in curr_vocab):
        return word
    else:
        return '<unk>'

def gen_vocab(curr_vocab, vocab_size, predefined, outfname):
    if(vocab_size is None):
        vocab_size = len(curr_vocab + predefined)
    vocab_to_include = predefined + [x[0] for x in curr_vocab.most_common(vocab_size - len(predefined))]
    idx2tok = []
    with open(outfname, mode='wt', encoding='utf-8') as f:
        for item in vocab_to_include:
            f.write(item + '\n')
            idx2tok.append(item)
    tok2idx = {tok:i for i, tok in enumerate(idx2tok)}
    return idx2tok, tok2idx
        
def pad(seq_list, max_length):
    seqlen = len(seq_list)
    seq_list += ['<eos>'] * (max_length - seqlen)
    return seq_list

def load_vocab(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        idx2tok = [x[:-1] for x in f.readlines()][:-1]
    tok2idx = {tok:i for i, tok in enumerate(idx2tok)}
    return idx2tok, tok2idx

##############################################

print("Loading raw train/valid/test data...")
with open(args.train_input_fname, 'rb') as f:
    raw_train_data = pickle.load(f)
random.shuffle(raw_train_data)
with open(args.valid_input_fname, 'rb') as f:
    raw_valid_data = pickle.load(f)
with open(args.test_input_fname, 'rb') as f:
    raw_test_data = pickle.load(f)

print("Generating model vocabulary...")
vocab = Counter()
aux_vocab = Counter()
for sent in tqdm(raw_train_data + raw_valid_data + raw_test_data):
    vocab.update([lower_num(word[0]) for word in sent])
    aux_vocab.update([word[1] for word in sent])

vocab_idx2tok, vocab_tok2idx = gen_vocab(vocab, args.vocab_size, ['<eos>', '<unk>'], args.vocab_fname)
aux_vocab_idx2tok, aux_vocab_tok2idx = gen_vocab(aux_vocab, args.aux_vocab_size, ['<eos>'], args.aux_vocab_fname)
##############################################
print('Generating datasets...')
train = h5py.File(args.train_input_fname + '.hdf5', 'w')
valid = h5py.File(args.valid_input_fname + '.hdf5', 'w')
test = h5py.File(args.test_input_fname + '.hdf5', 'w')
    
for f, raw_data in [(train, raw_train_data), (valid, raw_valid_data), (test, raw_test_data)]:
    input_toks = f.create_dataset('input_toks', (len(raw_data), args.seq_len), dtype=int)
    aux_labels = f.create_dataset('aux_labels', (len(raw_data), args.seq_len), dtype=int)
    seq_lens = f.create_dataset('length', (len(raw_data),), dtype=int)

    for i, sent_i in tqdm(enumerate(raw_data), total=len(raw_data)):
        sent_i = sent_i[:args.seq_len-1]
        input_toks[i] = [vocab_tok2idx[x] for x in pad(['<eos>'] + [unk(lower_num(word_j[0]), vocab_tok2idx) for word_j in sent_i], args.seq_len)]
        aux_labels[i] = [aux_vocab_tok2idx[x] for x in pad(['<eos>'] + [unk(word_j[1], aux_vocab_tok2idx) for word_j in sent_i], args.seq_len)]
        seq_lens[i] = len(sent_i)
    f.close()
print('HDF5 file created.')