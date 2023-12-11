'''
Code for training and evaluating a neural language model.
LM can output incremental complexity measures and be made adaptive.
'''

from __future__ import print_function
import argparse
import time
import math
import numpy as np
import sys
import warnings
import torch
import torch.nn as nn
import data
import model
import pickle
import os.path

import torch.nn.functional as F
from torch import Tensor

# suppress SourceChangeWarnings
warnings.filterwarnings("ignore")

sys.stderr.write('Libraries loaded\n')

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')

# Model parameters
parser.add_argument('--model', type=str, default='CBRRNN',
                    choices=['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU','CBRRNN'],
                    help='type of recurrent net')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--init', type=float, default=None,
                    help='-1 to randomly Initialize. Otherwise, all parameter weights set to value')

#Model objective parameters
parser.add_argument('--objective', type=str, default='lm',
                    choices=['lm', 'lc_parsing', 'dep_parsing'],
                    help='what the model is modeling. "lm" is word sequences, "lc_parsing" and "dep_parsing" are left corner const. and dependency parsing actions, respectively')
parser.add_argument('--scale_loss_by_wordtype', action='store_true',
                    help='scale loss of certain token types (useful for scaling action and terminal loss in parsing)')
parser.add_argument('--aux_objective', action='store_true',
                    help='use flag to train on an auxillary objective like CCG supertagging')
parser.add_argument('--aux_objective_weight', type=float, default=5.0,
                    help='relative weight of aux. objective when calculataing loss: lm_loss + w*aux_loss')
parser.add_argument('--ablate_attention', action='store_true',
                    help='use flag to ablate the model attention mechanism of a CBRNN during training')
parser.add_argument('--uniform_attention', action='store_true',
                    help='use flag to make the attention distribution uniform - ie. a non weighted average of all non masked values')
parser.add_argument('--attn_entropy_loss', action='store_true',
                    help='Use an attention entropy loss term')
parser.add_argument('--attn_entropy_loss_weight', type=float, default=0.1,
                    help='weight for entropy loss term')
parser.add_argument('--attn_entropy_loss_delay', type=int, default=5,
                    help='Number of epochs to train before incorporating attention entropy loss')
parser.add_argument('--force_overwrite', action='store_true',
                    help='force model to save params from latest epoch even if not the best loss')
#parser.add_argument('--dep_retrieval_loss', action='store_true',
#                    help='train a model to do incremental dependency parsing through self-attention')

# Data parameters
parser.add_argument('--model_file', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--data_dir', type=str, default='./data/wikitext-2',
                    help='location of the corpus data')
parser.add_argument('--vocab_file', type=str, default='vocab.txt',
                    help='path to load/save the vocab file')
parser.add_argument('--aux_vocab_file', type=str, default='aux_vocab.txt',
                    help='path to load/save the aux. label vocab file')
parser.add_argument('--embedding_file', type=str, default=None,
                    help='path to pre-trained embeddings')
parser.add_argument('--trainfname', type=str, default='train.txt',
                    help='name of the training file')
parser.add_argument('--validfname', type=str, default='valid.txt',
                    help='name of the validation file')
parser.add_argument('--testfname', type=str, default='test.txt',
                    help='name of the test file')

# Runtime parameters
parser.add_argument('--test', action='store_true',
                    help='test a trained LM')
parser.add_argument('--load_checkpoint', action='store_true',
                    help='continue training a pre-trained LM')
parser.add_argument('--freeze_embedding', action='store_true',
                    help='do not train embedding weights')
parser.add_argument('--log_interval', type=int, default=500, metavar='N',
                    help='report interval')

args = parser.parse_args()

    
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    print("CUDA device availible")
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        print("using CUDA")
        torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
torch.autograd.set_detect_anomaly(True)

###############################################################################
# Load data
###############################################################################
print('Loading data...')
if (args.aux_objective):
    aux_vocab_file = args.aux_vocab_file
else:
    aux_vocab_file = None

corpus = data.SentenceCorpus(args.data_dir, args.vocab_file, device,
                                model_type=args.model,
                                trainfname=args.trainfname,
                                validfname=args.validfname,
                                testfname=args.testfname,
                                batch_size=args.batch_size,
                                test_flag=args.test,
                                seed=args.seed,
                                aux_objective_flag=args.aux_objective,
                                aux_vocab_file=aux_vocab_file)
    

print('Done!\nBuilding model...')
###############################################################################
# Build/load the model
###############################################################################

if not args.test:
    if args.load_checkpoint:
        # Load the best saved model.
        print(' Continuing training from previous checkpoint')
        with open(args.model_file, 'rb') as f:
            if args.cuda:
                model = torch.load(f).to(device)
            else:
                model = torch.load(f, map_location='cpu')
            if not hasattr(model, "uniform_attention"): #to handle "legacy" models
                model.uniform_attention = False
    else:
        ntokens = len(corpus.dictionary)
        if(args.aux_objective):
            nauxclasses = len(corpus.aux_dictionary)
        else:
            nauxclasses = 0
        if(args.model != 'CBRRNN'):
            model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                               args.nlayers, embedding_file=args.embedding_file,
                               dropout=args.dropout, tie_weights=args.tied,
                               freeze_embedding=args.freeze_embedding, 
                               aux_objective=args.aux_objective,
                               nauxclasses=nauxclasses).to(device)
        else:
            model = model.CueBasedRNNModel(ntokens, args.emsize, args.nhid,
                               args.nlayers, embedding_file=args.embedding_file,
                               dropout=args.dropout, tie_weights=args.tied,
                               freeze_embedding=args.freeze_embedding, 
                               aux_objective=args.aux_objective,
                               nauxclasses=nauxclasses, 
                               ablate_attention=args.ablate_attention, 
                               uniform_attention = args.uniform_attention, device=device).to(device)

 
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    if(args.model != 'CBRRNN'):
        model.rnn.flatten_parameters()
    # setup model with optimizer and scheduler
criterion = nn.CrossEntropyLoss(reduction='none')

total_params = 0
for parameter in model.parameters():
    if len(parameter.shape) > 1:
        curr_param_num = 1
        for dim in parameter.shape:
            curr_param_num *= dim
    else:
        curr_param_num = parameter.shape[0]
    total_params += curr_param_num
print("Done!")
print("Total parameters of model: ", total_params)
if(not args.test):
    print("Beginning Training...")
sys.stdout.flush()

###############################################################################
# Training code
###############################################################################

def log_training_step(loss_data, epoch, elapsed, lr, batch_num=None, total_batches=None, train=True, combined_loss=None, aux_acc=None):
    if(train):
        outstr = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f}'.format(
                                epoch, batch_num, total_batches, lr, elapsed * 1000 / args.log_interval)
        for loss_type, loss_value in loss_data.items():
            loss_mean = loss_value[0] / loss_value[1]
            outstr += ' | {} loss {:5.3f} | {} ppl {:8.3f}'.format(loss_type, loss_mean, loss_type, math.exp(loss_mean))
    else:
        outstr = ('-' * 89) + '\n'
        outstr += '| end of epoch {:3d} | time: {:5.2f}s | lr: {:4.8f}'.format(epoch, elapsed, lr)
        for loss_type, loss_value in loss_data.items():
            loss_mean = loss_value[0] / loss_value[1]
            outstr += ' | valid {} loss {:5.3f} | valid {} ppl {:8.3f}'.format(loss_type, loss_mean, loss_type, math.exp(loss_mean))
        if(aux_acc is not None):
            outstr += ' | valid aux acc {:8.3f}'.format((aux_acc / loss_data['lm'][1]) * 100)
        outstr += ' | combined loss {:8.9f} \n'.format(combined_loss / loss_data['lm'][1])
        outstr += ('-' * 89)
    print(outstr)
    sys.stdout.flush()


def get_loss(output_data, all_data, curr_example, aux_preds, loss_types):
    losses = {}
    output_data = output_data.transpose(1,2)
    target_data = all_data.input_toks[curr_example][1:]
    if(all_data.lm_loss_pad_masks is not None):
        length_masks = all_data.lm_loss_pad_masks[curr_example][1:]
    else:
        length_masks = None

    lm_loss = criterion(output_data, target_data)
    if(length_masks is not None):
        lm_loss = lm_loss * length_masks
        total_lm_exs = length_masks.sum().item()
    else:
        total_lm_exs = target_data.nelement().item()
    losses['lm'] = (lm_loss, total_lm_exs)
    if(aux_preds is not None and ('aux' in loss_types)):
        aux_target_data = all_data.aux_labels[curr_example][:-1]
        aux_loss = criterion(aux_preds.transpose(1,2), aux_target_data)
        if(length_masks is not None):
             aux_loss = aux_loss * length_masks
        losses['aux'] = (aux_loss, total_lm_exs)
    if((all_data.type_loss_masks is not None) and ('terminal' in loss_types)): #todo - deal with logging without backpropping this loss
        type_masks = all_data.type_loss_masks[curr_example][1:]
        losses['terminal'] = ((lm_loss * type_masks), type_masks.sum().item())
        losses['nt'] = ((lm_loss * (~type_masks)), (length_masks * (~type_masks)).sum().item())
    return losses

def scale_loss(loss_data):
    if(args.scale_loss_by_wordtype):
        t_loss_mean = loss_data['terminal'][0].sum() / loss_data['terminal'][1]
        nt_loss_mean = loss_data['nt'][0].sum() / loss_data['nt'][1]
        lm_loss = ((1 / (1+args.terminal_loss_weight)) * t_loss_mean) + ((args.terminal_loss_weight / (1+ args.terminal_loss_weight)) * nt_loss_mean)
    else:
        lm_loss = loss_data['lm'][0].sum() / loss_data['lm'][1]

    if(args.aux_objective):
        aux_loss = loss_data['aux'][0].sum() / loss_data['aux'][1]
        loss = ((1 / (1+ args.aux_objective_weight)) * lm_loss) + ((args.aux_objective_weight / (1+ args.aux_objective_weight)) * aux_loss)
    else:
        loss = lm_loss

    return loss

def get_acc(output_data, all_data, current_ex):
    targets = all_data.aux_labels[current_ex]
    preds = output_data.argmax(1, dim=-1)
    length_masks = all_data.lm_loss_pad_masks[current_ex][:-1]
    total_correct = ((targets == preds) * length_masks).sum().item()
    return total_correct

def get_attn_entropy_loss(attn_weights):
    entropies = []
    for length_i_dists in attn_weights[2:]:
        length_i_entropy = -(length_i_dists * (length_i_dists + .000000001).log2()).sum(axis=1) / np.log2(length_i_dists.shape[-1])
        length_i_entropy = length_i_entropy.mean() / len(attn_weights[2:]) 
        entropies.append(length_i_entropy)
    return entropies

def repackage_hidden(in_state):
    """ Wraps hidden states in new Tensors, to detach them from their history. """
    if isinstance(in_state, torch.Tensor):
        return in_state.detach()
    else:
        return tuple(repackage_hidden(value) for value in in_state)

def evaluate(data_source):
    """ Evaluate for validation (no adaptation, no complexity output) """
    # Turn on evaluation mode which disables dropout.
    if(args.objective == 'lm'):
        total_loss = {'lm':[0.0, 0]}
    elif(args.objective == 'lc_parsing' or args.objective == 'dep_parsing'):
        total_loss = {'lm':[0.0, 0], 'terminal':[0.0, 0], 'nt':[0.0, 0]}
    if(args.aux_objective):
        total_aux_correct = 0.0
        total_loss.update({'aux':[0.0, 0]})
    total_scaled_loss = 0.0
    model.eval()
    with torch.no_grad():
        # Construct hidden layers for each sub-batch
        if(args.model != 'CBRRNN'):
            hidden_batch = model.init_hidden(args.batch_size)
        for batch in range(len(data_source)):
            batch_data = data_source.input_toks[batch]
            if(args.model == 'CBRRNN'):
                batch_masks = data_source.input_masks[batch]
                cache = model.init_cache(batch_data)
                output, _, aux_preds, _ = model(batch_data[:-1], cache, masks=batch_masks[:,:-1]) 
            else:
                output, hidden_batch, aux_preds = model(batch_data[:-1], hidden_batch)
            loss_data = get_loss(output, data_source, batch, aux_preds, total_loss.keys())
            for loss_type_i in loss_data.keys():
                total_loss[loss_type_i][0] += loss_data[loss_type_i][0].sum().item()
                total_loss[loss_type_i][1] += loss_data[loss_type_i][1]
            if(args.aux_objective):
                total_aux_correct += get_acc(aux_preds, data_source, batch)
            else:
                total_aux_correct = None

            total_scaled_loss += scale_loss(loss_data)
    return total_scaled_loss, total_loss, total_aux_correct

def train(epoch):
    """ Train language model """
    # Turn on training mode which enables dropout.
    model.train()
    #losses are tracked in dicts. of tuples
    if(args.objective == 'lm'):
        total_loss = {'lm':[0.0, 0]}
    elif(args.objective == 'lc_parsing' or args.objective == 'dep_parsing'):
        total_loss = {'lm':[0.0, 0], 'terminal':[0.0, 0], 'nt':[0.0, 0]}
    if(args.aux_objective):
        total_loss.update({'aux':[0.0, 0]})
    start_time = time.time()
    true_batch_num = 0
    hidden_batch = []
    #if model is RNN, then pre-initialize hidden states for the batch, Cue-based model does this in the forward pass
    if(args.model != 'CBRRNN'):
        hidden_batch= model.init_hidden(args.batch_size)
    curr_train_chunk = corpus.get_train_chunk()
    if(args.objective == 'lm'):
        lm_input_mask = torch.ones(curr_train_chunk.input_toks[0].shape[-1], len(curr_train_chunk.input_toks[0]), len(curr_train_chunk.input_toks[0]) + 1)
        lm_input_mask[:,0,1:] = 0
        lm_input_mask[:,1:,0] = 0
        lm_input_mask[:,2:,1] = 0
        lm_input_mask = lm_input_mask.log()
    while(curr_train_chunk is not None): 
        for batch in range(len(curr_train_chunk)):
            batch_data = curr_train_chunk.input_toks[batch]
            if(args.model == 'CBRRNN'):
                if(curr_train_chunk.input_masks is not None):
                    batch_masks = curr_train_chunk.input_masks[batch][:,:-1]
                else:
                    batch_masks = lm_input_mask
                cache = model.init_cache(batch_data)
                output, hidden_batch_all, aux_preds, _ = model(batch_data[:-1], cache, masks=batch_masks, output_attn=args.attn_entropy_loss)
                hidden_batch = hidden_batch_all[-1]
            else:
                batch_masks = None
                output, hidden_batch, aux_preds = model(batch_data, hidden_batch)

            loss_data = get_loss(output, curr_train_chunk, batch, aux_preds, total_loss.keys())
            for loss_type_i in loss_data.keys():
                total_loss[loss_type_i][0] += loss_data[loss_type_i][0].sum().item()
                total_loss[loss_type_i][1] += loss_data[loss_type_i][1]

            loss = scale_loss(loss_data)
                
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            # Detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            if(args.model != 'CBRRNN'):
                hidden_batch = repackage_hidden(hidden_batch)
            model.zero_grad()

            if true_batch_num % args.log_interval == 0 and true_batch_num > 0:
                elapsed = time.time() - start_time
                log_training_step(total_loss, epoch, elapsed, float(optimizer.param_groups[0]['lr']), batch_num=true_batch_num, total_batches=int(corpus.num_train_exs/args.batch_size), train=True)
                total_loss = {k:[0.0, 0] for k in total_loss.keys()}
                start_time = time.time()
            true_batch_num += 1
        curr_train_chunk = corpus.get_train_chunk()


#load checkpoint training config data
checkpoint_train_config_fname = args.model_file + '.config'
if(os.path.isfile(checkpoint_train_config_fname)):
    with open(checkpoint_train_config_fname, 'rb') as f:
        checkpoint_config = pickle.load(f)
    best_val_loss = checkpoint_config['best_val_loss']
    no_improvement = checkpoint_config['no_improvement']
    start_epoch = checkpoint_config['start_epoch']
    if(args.lr == 1.0): #default
        lr = checkpoint_config['lr']
    else:
        lr = args.lr
else:
    best_val_loss = None
    no_improvement = 0
    start_epoch = 1
    lr = args.lr

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=0,factor=0.1)
if(best_val_loss is not None and not args.force_overwrite):
    scheduler.step(best_val_loss)

# At any point you can hit Ctrl + C to break out of training early.
if not args.test:
    try:
        for epoch in range(start_epoch, args.epochs+1):
            epoch_start_time = time.time()
            train(epoch)
            combined_loss, all_loss, aux_acc = evaluate(corpus.valid)
            log_training_step(all_loss, epoch, (time.time() - epoch_start_time), lr, train=False, combined_loss=combined_loss, aux_acc=aux_acc)
            # Save the model if the validation loss is the best we've seen so far.
            if (not best_val_loss or (combined_loss < best_val_loss)) or (args.force_overwrite) or (epoch == args.attn_entropy_loss_delay):
                no_improvement = 0
                with open(args.model_file, 'wb') as f:
                    torch.save(model, f)
                    best_val_loss = combined_loss
                args.force_overwrite = False
            else:
                # Anneal the learning rate if no more improvement in the validation dataset.
                no_improvement += 1
                if no_improvement >= 3:
                    print('Covergence achieved! Ending training early')
                    break
            if(epoch == args.attn_entropy_loss_delay): #re-initialize scheduler
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=0,factor=0.1)
            scheduler.step(combined_loss)
            checkpoint_config = {'best_val_loss':best_val_loss, 'no_improvement':no_improvement,'start_epoch':epoch+1,'lr':lr}
            with open(checkpoint_train_config_fname, 'wb') as config_f:
                pickle.dump(checkpoint_config, config_f)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
