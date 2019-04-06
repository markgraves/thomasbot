import unicodedata
import string
import random

import torch
import torch.autograd

ALL_LETTERS = string.ascii_letters + " .,;'-"

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

# Read a file and split into lines
def read_lines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


class Corpus():
    def __init__(self, filename=None):
        self.filename = filename if filename is not None else ''
        self.all_letters = ALL_LETTERS

        self.lines = []
        self.num_lines = 0

    def load(self):
        if self.filename:
            self.lines = read_lines(self.filename)
            self.num_lines = len(self.lines)
        else:
            print('No file to load.')

    def num_letters(self):
        return len(self.all_letters)

    def random_line(self):
        return self.lines[random.randint(0, self.num_lines -1)]

    def random_line_tensor(self):
        return line_to_tensor(self.random_line(), self.all_letters)

    def random_line_variable(self):
        return torch.autograd.Variable(self.random_line_tensor())


def fetch_aquinas_corpus():
    corpus = Corpus('../../data/aquinas/aquinas-summa-pt2-q-stripped.txt')
    corpus.load()
    return corpus


# Turn a line into a <line_length x 1 x num_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line, all_letters):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor
