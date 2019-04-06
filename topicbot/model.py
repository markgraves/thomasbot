import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, width_classifier=0):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.width_classifier = width_classifier
        if width_classifier > 0:
            # self.classifier = nn.Linear(width_classifier, ntoken)  # bug w args, prob not needed
            self.classifier = nn.Dropout(0)
        else:
            self.classifier = None
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp + width_classifier, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp + width_classifier, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        #if self.classifier is not None:
        #    self.classifier.bias.data.fill_(0)
        #    self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, classification=None):
        emb = self.drop(self.encoder(input))
        if self.classifier is not None and classification is not None:  # may need to change this later
            # trans_classification = self.drop(self.classifier(classification))
            # trans_classification = self.classifier(classification)
            if len(classification.size()) < 3:  # needed if only one dimension of cdata
                trans_classification = classification.unsqueeze(2)
            else:
                trans_classification = classification
            # print(emb.size())
            # print(trans_classification.size())
            try:
                combined = torch.cat((emb, trans_classification), 2)
            except RuntimeError:
                print('Internal Error: Size mismatch while combining in model', emb.size(), trans_classification.size())
                raise
            output, hidden = self.rnn(combined, hidden)
        else:
            output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
