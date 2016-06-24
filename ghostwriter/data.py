from itertools import cycle

from numpy import array_split
from numpy.core.numeric import array


class LabeledData(object):
    """
    A set of vectors and their associated labels.
    """

    def __init__(self, vectors, labels):
        assert len(vectors) == len(labels)
        self.vectors = array(vectors)
        self.labels = array(labels)

    def __repr__(self):
        return "Labeled data, %d instances" % len(self)

    def __len__(self):
        return len(self.vectors)

    def batches(self, batch_size):
        """
        Iterate repeatedly through the data, returning batches of the specified size.

        :param batch_size: number of instances per batch
        :type batch_size: int
        :return: vectors and labels
        :rtype: (numpy.array, numpy.array)
        """
        vector_batches = array_split(self.vectors, batch_size)
        label_batches = array_split(self.labels, batch_size)
        return zip(cycle(vector_batches), cycle(label_batches))


def skip_gram_data_set(tokens, width):
    targets, contexts = zip(*skip_gram(tokens, width))
    return LabeledData(targets, contexts)


def skip_gram(tokens, width):
    """
    Generate skip grams from a sequence of tokens.

    :param tokens: sequence of tokens
    :type tokens: iterator
    :param width: context window width
    :type width: int
    :return: target token, context token pairs
    :rtype: iterator of (str, str)
    """
    window = []
    for token in tokens:
        window.append(token)
        n = len(window)
        if n == 2 * width + 1:
            m = n // 2
            target = window[m]
            for context in window[:m]:
                yield target, context
            for context in window[m + 1:]:
                yield target, context
            window.pop(0)
