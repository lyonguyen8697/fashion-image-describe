import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import string


class Vocabulary(object):
    def __init__(self, size, end_token, save_file=None):
        self.words = []
        self.word2idx = {}
        self.word_frequencies = []
        self.end_token = end_token
        self.size = size
        if save_file is not None:
            self.load(save_file)

    def build(self, sentences):
        """ Build the vocabulary and compute the frequency of each word. """
        word_counts = {}
        for sentence in tqdm(sentences):
            for w in sentence.lower().split():
                word_counts[w] = word_counts.get(w, 0) + 1.0

        assert self.size-1 <= len(word_counts.keys())
        self.words.append('<start>')
        self.word2idx['<start>'] = 0
        self.word_frequencies.append(1.0)

        word_counts = sorted(list(word_counts.items()),
                            key=lambda x: x[1],
                            reverse=True)

        for idx in range(self.size-1):
            word, frequency = word_counts[idx]
            self.words.append(word)
            self.word2idx[word] = idx + 1
            self.word_frequencies.append(frequency)

        self.word_frequencies = np.array(self.word_frequencies)
        self.word_frequencies /= np.sum(self.word_frequencies)
        self.word_frequencies = np.log(self.word_frequencies)
        self.word_frequencies -= np.max(self.word_frequencies)

    def process_sentence(self, sentence):
        """ Tokenize a sentence, and translate each token into its index
            in the vocabulary. """
        words = sentence.lower().split()
        word_idxs = [self.word2idx[w] for w in words]
        return word_idxs

    def get_sentence(self, idxs):
        """ Translate a vector of indicies into a sentence. """
        words = [self.words[i] for i in idxs]

        if self.end_token in words:
            length = np.argmax(np.array(words) == self.end_token) + 1
        else:
            length = len(words)
        words = words[:length]

        sentence = ' '.join(words)

        return sentence

    def save(self, save_file):
        """ Save the vocabulary to a file. """
        data = pd.DataFrame({'word': self.words,
                             'index': list(range(self.size)),
                             'frequency': self.word_frequencies})
        data.to_csv(save_file)

    def load(self, save_file):
        """ Load the vocabulary from a file. """
        assert os.path.exists(save_file)
        data = pd.read_csv(save_file)
        self.words = data['word'].values
        self.word2idx = {self.words[i]: i for i in range(self.size)}
        self.word_frequencies = data['frequency'].values
