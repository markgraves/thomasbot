import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


class CharLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=0.2)

        # The linear layer that maps from hidden state space to output space
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = autograd.Variable(torch.zeros(self.layers, 1, self.hidden_dim))

    def forward(self, char):
        lstm_out, self.hidden = self.lstm(char, self.hidden)
        out_space = self.hidden2out(lstm_out.view(len(sentence), -1))
        out_scores = F.log_softmax(out_space)
        return out_scores


class LSTMTopic(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim, topic_dim):
        super(LSTMTopic, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2, dropout=0.2)

        # The linear layer that maps from hidden state space to output space
        self.hidden2out = nn.Linear(hidden_dim, output_dim)

        # The linear layer that maps from hidden state space to topic space
        self.hidden2topic = nn.Linear(hidden_dim, topic_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        out_space = self.hidden2out(lstm_out.view(len(sentence), -1))
        out_scores = F.log_softmax(out_space)
        topic_space = self.hidden2topic(lstm_out.view(len(sentence), -1))
        topic_scores = F.log_softmax(topic_space)
        # concat out and topic scores
        return torch.cat(out_scores, topic_scores)
