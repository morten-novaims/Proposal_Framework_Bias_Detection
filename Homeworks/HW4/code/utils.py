from copy import deepcopy
import numpy as np

def logzero():
    return -np.inf

class stringdict():
    def __init__(self, d=None):
        if d is None:
            d = dict([('unk',0)])
        self._wdict = d
        self._names = list(d.keys())
        self._freeze = False

    def add(self, word):
        if not self._freeze:
            id = self._wdict.get(word, len(self._wdict))
            if word not in self._wdict:
                self._names.append(word)
            self._wdict[word] = id
        else:
            id = self._wdict.get(word, 0)
        return id

    def get_id(self, word):
        return self._wdict.get(word, 0)

    def get_word(self, id):
        if id > len(self._names):
            return 'unk'
        return self._names[id]

    def freeze(self):
        self._freeze = True

    def unfreeze(self):
        self._freeze = False

    @property
    def len(self):
        return len(self._wdict)

    @property
    def dict(self):
        return self._wdict  

    @property
    def names(self):
        return self._names



class Seq():
    def __init__(self, words, labels):
        self._words = words
        self._labels = labels
        assert len(words)==len(labels)

    @property
    def x(self):
        """
        :return: the words in the sequence
        """
        return self._words

    @property
    def y(self):
        """
        :return: the labels in the sequence
        """
        return self._labels

    @property
    def len(self):
        return len(self._words)

    def copy(self):
        return Seq(deepcopy(self._words), deepcopy(self._labels))

