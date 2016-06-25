import operator

import numpy
from numpy import array


def labeled_data(labeled_data_epoch):
    """
    This enumerates repeatedly over labeled data, returning pairs of vectors and labels.

    :param labeled_data_epoch: function that returns an iterator over all the vectors and labels in an epoch of data
    :type labeled_data_epoch: function
    :return: epoch number, iteration number, vectors, labels
    :rtype: (int, int, numpy.array, numpy.array)
    """
    epoch = 1
    iteration = 1
    while True:
        for xs, ys in labeled_data_epoch():
            yield epoch, iteration, xs, ys
            iteration += 1
        epoch += 1


def sequence_data_epoch(items, batch_size, sequence_length):
    """
    Enumerator over sequences of items, as in the training data for a language model.

    :param items: the items to enumerate over
    :type items: iterable
    :param batch_size:
    :type batch_size: int
    :param sequence_length:
    :type sequence_length: int
    :return: enumeration over items and their preceding contexts
    :rtype: iteration of (numpy.array, numpy.array), each array is of size batch_size x sequence_length
    """

    def batched_sequences(batch_items):
        zs = numpy.zeros(n, dtype=items.dtype)
        numpy.copyto(zs[:len(batch_items)], batch_items)
        return zs.reshape((batch_size, sequence_length))

    n = batch_size * sequence_length
    for i in range(0, len(items), n):
        xs = batched_sequences(items[i:i + n])
        ys = batched_sequences(items[i + 1:i + n + 1])
        yield xs, ys


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
        vector_batches = batch_iterate(self.vectors, batch_size, concatenate_arrays)
        label_batches = batch_iterate(self.labels, batch_size, concatenate_arrays)
        return zip(vector_batches, label_batches)


def concatenate_arrays(a, b):
    return numpy.concatenate((a, b))


def batch_iterate(xs, batch_size, concatenate=operator.add):
    """
    Iterate over a sequence yielding batches of a specified size. When the end of the sequence is reached, start over at
    the beginning. The batches will all be the same size and the iterator is infinite.

    The sequence concatenation can be specified, since different sequence types may have different concatenation
    operations.

    >>> batch_iterate("abcd", 3) # Generates ["abc", "dab", "cda", "bcd", "abc", "dab" ...]

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
