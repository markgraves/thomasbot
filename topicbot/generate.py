import argparse
import os, sys
import contextlib

import torch
from torch.autograd import Variable
import sentencepiece as spm

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='',  #deprecated
                    help='location of the data corpus')
parser.add_argument('--datacorpus', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='',
                    help='location of the data corpus vocab')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='torch model checkpoint to use')
parser.add_argument('--bpemodel', type=str, default='./bpe.model',
                    help='bpe model to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--outfraw', type=str, default='',
                    help='output file for generated text in raw (bpe) format')
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

device = torch.device("cuda" if args.cuda else "cpu")

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

sp = spm.SentencePieceProcessor()
sp.load_from_serialized_proto(open(args.bpemodel, 'rb').read())

torch_model_basepath = os.path.splitext(args.checkpoint)[0]
if not args.vocab:
    args.vocab = torch_model_basepath + '_vocab.json'
    if not os.path.exists(args.vocab):
        print('Warning: Model vocab file does not exist: ' + args.vocab)
        args.vocab = ''

if args.vocab:
    dictionary = data.Dictionary.load_dictionary(args.vocab)
elif args.datacorpus:
    corpus = data.Corpus(args.datacorpus)
    dictionary = corpus.dictionary
elif args.data:
    corpus = data.Corpus(args.data)
    dictionary = corpus.dictionary
else:
    print('Error: need vocabulary or corpus to obtain tokens')
    sys.exit(1)

ntokens = len(dictionary)
hidden = model.init_hidden(1)
try:
    width_classifier = model.width_classifier
except AttributeError:
    width_classifier = model.width_classifer  # misspelling needed for models prior to 1/20/18
input = torch.randint(ntokens, (1, 1), dtype=torch.long)
if args.cuda:
    input.data = input.data.cuda()

end_tags = set(['</s>', '<eos>', '</question>', '</article>'])

@contextlib.contextmanager
def noop():
    yield None

with open(args.outfraw, 'w') if args.outfraw else noop() as outfraw:
    with open(args.outf, 'w') as outf:
        with torch.no_grad():
            text = []
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
                word_weights = output.squeeze().data.div(args.temperature).exp().cpu()   #per pytorch bug GH-322
                # word_weights = output.squeeze().detach().div(args.temperature).exp().cpu()
                prediction = output.squeeze().detach()
                prediction = prediction.div(args.temperature)
                # to prevent overflow problem with small temperature values, substract largest value from all
                # this makes a vector in which the largest value is 0
                max = torch.max(prediction)
                prediction -= max
                word_weights = prediction.exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.data.fill_(word_idx)
                word = dictionary.idx2word[word_idx]
                if word not in end_tags:
                    if outfraw:
                        outfraw.write(word + ' ')
                    text.append(word)
                else:
                    if outfraw:
                        outfraw.write(word + '\n')
                    decoded_text = sp.decode_pieces(text)
                    if decoded_text.strip():
                        outf.write(decoded_text)
                        outf.write('\n')
                    text = []
                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))
        if text:
            outf.write(sp.decode_pieces(text))
        outf.write('\n')
        if outfraw:
            outfraw.write('\n')
