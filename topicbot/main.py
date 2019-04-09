# coding: utf-8
import argparse
import sys
import os
import time
import math
import numpy as np
import json
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--lrdecay', type=float, default=4,
                    help='learning rate decay')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--output_loss', type=str, default='loss_data.json',
                    help='file to save the loss data')
parser.add_argument('--output_vocab', type=str, default='',
                    help='file to save the vocabulary')
parser.add_argument('--topicnum', type=int, default=-1,
                    help='topic number (default -1 for none, 0 for all)')
args = parser.parse_args()

save_basepath = os.path.splitext(args.save)[0]
if not args.output_vocab:
    args.output_vocab = save_basepath + '_vocab.json'
if not args.output_loss:
    args.output_loss = save_basepath + '_loss.json'

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
corpus.save_dictionary(args.output_vocab)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def batchify_cdata(data, bsz):  # note: function only works on contiguos data, due to view
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    ndatadims = data.size(1)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = np.transpose(data.view(bsz, -1, ndatadims), [1,0,2]).contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
if not corpus.hascdata or args.topicnum == -1:  # no cdata
    train_cdata = None
    val_cdata = None
    test_cdata = None
    numcdatadims = 0
elif args.topicnum == 0:  # all cdata
    train_cdata = batchify_cdata(corpus.traincdata, args.batch_size)
    val_cdata = batchify_cdata(corpus.validcdata, eval_batch_size)
    test_cdata = batchify_cdata(corpus.testcdata, eval_batch_size)
    numcdatadims = corpus.datamaxwidth
elif args.topicnum <= corpus.traincdata.size(1):  # specific topic, note: begins with 1, not 0
    train_cdata = batchify(corpus.traincdata[:,args.topicnum].contiguous(), args.batch_size)
    val_cdata = batchify(corpus.validcdata[:,args.topicnum].contiguous(), eval_batch_size)
    test_cdata = batchify(corpus.testcdata[:,args.topicnum].contiguous(), eval_batch_size)
    numcdatadims = 1
else:  # invalid arg
    print('Error: Invalid number of topics %s. Must be <= %s' % (args.topicnum, corpus.traincdata.size(1)))
    sys.exit()

print('Building model on data (dims): %s' % list(train_data.size()))
if train_cdata is not None:
    print('Model includes classificaion data (dims): %s' % (train_cdata.size(2) if len(train_cdata.size()) > 2 else 1))

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, numcdatadims)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, source_cdata, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    if source_cdata is not None:
        cdata = source_cdata[i:i+seq_len]
    else:
        cdata = None
    target = source[i+1:i+1+seq_len].view(-1)
    return data, cdata, target


def evaluate(data_source, cdata_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, cdata, targets = get_batch(data_source, cdata_source, i)
            output, hidden = model(data, hidden, cdata)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    global ITERATION, LOSS_DATA
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, cdata, targets = get_batch(train_data, train_cdata, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden, cdata)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()
        ITERATION += 1

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            LOSS_DATA['train_loss_history'].append((epoch, batch, ITERATION, lr, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None
ITERATION = 0
LOSS_DATA = {'args': vars(args),
             'train_loss_history': [],
             'val_loss_history': [],
             'test_loss_history': [],
             'train_loss_header': ['epoch', 'batch', 'iteration', 'lr', 'train_loss', 'train_ppl'],
             'val_loss_header': ['epoch', 'iteration', 'lr', 'val_loss', 'val_ppl'],
             'test_loss_header': ['epoch', 'iteration', 'lr', 'test_loss', 'test_ppl'],
             'num_iterations': 0
}

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data, val_cdata)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | lr {:9.6f}'.format(epoch, (time.time() - epoch_start_time),
                                                        val_loss, math.exp(val_loss), lr))
        print('-' * 89)
        LOSS_DATA['val_loss_history'].append((epoch, ITERATION, lr, val_loss, math.exp(val_loss)))
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= args.lrdecay
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data, test_cdata)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
LOSS_DATA['test_loss_history'].append((epoch, ITERATION, lr, test_loss, math.exp(test_loss)))
LOSS_DATA['num_iterations'] = ITERATION

with open(args.output_loss, 'w') as f:
    json.dump(LOSS_DATA, f)
