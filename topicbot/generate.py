###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--topicnum', type=int, default=0,
                    help='topic number (default -1 for no topic, 0 for all)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    if args.cuda:
        model = torch.load(f)
    else:
        model = torch.load(f,map_location=lambda storage, loc: storage)

model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
try:
    width_classifier = model.width_classifier
except AttributeError:
    width_classifier = model.width_classifer  # misspelling needed for models prior to 1/20/18
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        if width_classifier == 0:
            output, hidden = model(input, hidden)
        else:
            if args.topicnum == -1:
                cdata = torch.zeros(1, 1, width_classifier)
            elif args.topicnum == 0:
                cdata = torch.rand(1, 1, width_classifier)
            elif args.topicnum > 0:
                cdata = torch.zeros(1, 1, width_classifier)
                if width_classifier == 1:
                    cdata[0][0][0] = 1  # topicnum could be actual topic num; not catching mismatch, then
                else:
                    cdata[0][0][args.topicnum - 1] = 1
            output, hidden = model(input, hidden, Variable(cdata))
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        outf.write(word + ('\n' if word in ['</s>', '<eos>'] else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
