import string
import os
import unicodedata
from collections import Counter
import numpy as np
import torch

ALL_LETTERS = string.ascii_letters + " .,;'-_</>" + chr(9601)  # last symbol is (U+2581) for sentence piece

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx2cnt = Counter()

    def add_word(self, word):
        if word in self.word2idx:
            self.idx2cnt[self.word2idx[word]] += 1
        else:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.idx2cnt[self.word2idx[word]] = 1            
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def num_tokens(self):
        return sum(self.idx2cnt.values())

    def compact(self, min_occurrences=0):
        new = Dictionary()
        newidx = 0
        for idx, word in enumerate(self.idx2word):
            if self.idx2cnt[idx] < min_occurrences:
                new.word2idx[word] = None
            else:
                new.idx2word.append(word)
                new.word2idx[word] = newidx
                new.idx2cnt[newidx] = self.idx2cnt[idx]
                newidx += 1
        return new

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.datacnt = 0
        self.datamaxwidth = 0
        if os.path.isdir(path):
            self.train, self.traincdata = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid, self.validcdata = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test, self.testcdata = self.tokenize(os.path.join(path, 'test.txt'))
        elif os.path.exists(path):
            self.train = self.tokenize(path)
            self.valid = None
            self.test = None
        else:
            print('Error: corpus file does not exist ' + path)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                if line.startswith('<data '):
                    self.datacnt += 1
                    parsed_data = line.strip()[11:-3]
                    num_items = parsed_data.count(',')  # +1 last -1 article_name
                    # num_items = 1 # STUB
                    if self.datamaxwidth == 0:
                        self.datamaxwidth = num_items
                    elif self.datamaxwidth != num_items:
                        print('Warning: number data items differs was %d now %d'
                              % (self.datamaxwidth, num_items))
                        self.datamaxwidth = max(self.datamaxwidth, num_items)
                    continue
                words = unicode_to_ascii(line).split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            if self.datacnt > 0:
                datavals = torch.FloatTensor(tokens, self.datamaxwidth)
                # datavals = torch.FloatTensor(tokens)  # STUB
            token = 0
            float_data = np.zeros(self.datamaxwidth)
            # float_data = 0.0  # STUB
            for line in f:
                if line.startswith('<data '):
                    parsed_data = line.strip()[11:-3]
                    split_data = parsed_data.split(',')
                    float_data = np.float32(split_data[1:])  # ignore article name
                    # float_data = np.float(split_data[5])  # STUB                    
                    continue
                words = unicode_to_ascii(line).split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    if self.datacnt > 0:
                        datavals[token] = torch.from_numpy(float_data)
                        # datavals[token] = float_data  # STUB
                    token += 1
        if self.datacnt == 0:
            self.hascdata = False
            return ids, None
        else:
            self.hascdata = True
            return ids, datavals
