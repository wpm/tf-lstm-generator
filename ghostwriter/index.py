from collections import Counter, defaultdict


class IndexedVocabulary(object):
    """
    Given a sequence of tokens, create a unique integer mapping from an integer to a type. Out of vocabulary types map
    to a default value of zero.

    If a maximum vocabulary size is specified, only the most frequent types will be indexed.
    """

    def __init__(self, tokens, max_vocabulary=None):
        """
        :param tokens: sequence of natural language tokens
        :type tokens: iterator of str
        :param max_vocabulary: maximum vocabulary size
        :type max_vocabulary: int or None
        """
        indexes = (t for t, _ in Counter(tokens).most_common(max_vocabulary))
        self.index_to_type = dict(enumerate(indexes, 1))
        # noinspection PyArgumentList
        self.type_to_index = defaultdict(int, dict(map(reversed, self.index_to_type.items())))

    def __repr__(self):
        return "Indexed Vocabulary, size %d" % len(self)

    def __str__(self):
        return "%s: %s ..." % (
            repr(self), " ".join("%s:%d" % (t, i) for i, t in sorted(self.index_to_type.items())[:5]))

    def __len__(self):
        return len(self.type_to_index)

    def index(self, type):
        return self.type_to_index[type]

    def type(self, index):
        return self.index_to_type[index]
