import operator

import numpy
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

        def concatenate(a, b):
            return numpy.concatenate((a, b))

        vector_batches = batch_iterate(self.vectors, batch_size, concatenate)
        label_batches = batch_iterate(self.labels, batch_size, concatenate)
        return zip(vector_batches, label_batches)


def batch_iterate(xs, batch_size, concatenate=operator.add):
    """
    Iterate over a sequence yielding batches of a specified size. When the end of the sequence is reached, start over at
    the beginning. The batches will all be the same size and the iterator is infinite.

    The sequence concatenation can be specified, since different sequence types may have different concatenation
    operations.

    >>> batch_iterate("abcd", 3), 5) # Generates ["abc", "dab", "cda", "bcd", "abc", "dab" ...]

    :param xs: items to group into batches
    :type xs: sequence
    :param batch_size: number of items in a batch
    :type batch_size: int
    :param concatenate: function to concatenate two sequences of items
    :type concatenate: func
    :return: iteration of item batches
    :rtype: iterator
    """
    n = len(xs)
    batch_size = min(batch_size, n)
    i = 0
    while True:
        j = (i + batch_size) % n
        if i < j:
            yield xs[i:j]
        else:
            yield concatenate(xs[i:], xs[:j])
        i = j


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
