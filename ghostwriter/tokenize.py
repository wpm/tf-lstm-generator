from collections import Counter
from itertools import chain

from nltk import word_tokenize


class MultiFileLineEnumerator(object):
    """
    Enumerate over all the lines in a sequence of files.

    File pointers are reset for each new iteration, so iteration is idempotent.
    """

    def __init__(self, text_files):
        """
        :param text_files: open file handles
        :type text_files: [file]
        """
        self.text_files = text_files

    def __repr__(self):
        return ", ".join(text_file.name for text_file in self.text_files)

    def __iter__(self):
        for text_file in self.text_files:
            text_file.seek(0)
        return chain(*(text_file.readlines() for text_file in self.text_files))


class Tokenizer(object):
    def __init__(self, text_lines):
        self.text_lines = text_lines


class EnglishWordTokenizer(Tokenizer):
    """
    Tokenize a sequence of text lines into English words
    """

    def __repr__(self):
        return "Word tokenizer for %s" % self.text_lines

    def __iter__(self):
        return chain(*(word_tokenize(line) for line in self.text_lines))


class CharacterTokenizer(Tokenizer):
    """
    Tokenize a sequence of text lines into characters.
    """

    def __repr__(self):
        return "Word tokenizer for %s" % self.text_lines

    def __iter__(self):
        return chain(*(iter(line) for line in self.text_lines))


class IndexedTokenizer(object):
    """
    Tokenize a sequence of text files into a set of integer indexes of the tokens.
    """

    def __init__(self, tokens, max_vocabulary=None):
        """
        :param tokens: iterator of tokens
        :type tokens: iterator
        :param max_vocabulary: maximum vocabulary size
        :type max_vocabulary: int or None
        """
        self.tokens = tokens
        self.indexed_vocabulary = IndexedVocabulary(self.tokens, max_vocabulary)

    def __repr__(self):
        return "%s, %s" % (self.tokens, self.indexed_vocabulary)

    def __iter__(self):
        return (self.indexed_vocabulary.index(token) for token in self.tokens)

    def __getitem__(self, index):
        """
        Look up a type in the vocabulary by its index

        :param index: type index
        :type index: int
        :return: type
        :rtype: str
        """
        return self.indexed_vocabulary.type(index)

    def vocabulary_size(self):
        return len(self.indexed_vocabulary)


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
        self.type_to_index = dict(map(reversed, self.index_to_type.items()))

    def __repr__(self):
        return "Indexed Vocabulary, size %d" % len(self)

    def __str__(self):
        return "%s: %s ..." % (
            repr(self), " ".join("%s:%d" % (t, i) for i, t in sorted(self.index_to_type.items())[:5]))

    def __len__(self):
        # Add 1 for the out of vocabulary type.
        return len(self.type_to_index) + 1

    def index(self, type):
        """
        :param type: a type listed in the vocabulary
        :type type: str
        :return: index of the type
        :rtype: int
        """
        return self.type_to_index.get(type, 0)

    def type(self, index):
        """
        :param index: index of a type in the vocabulary
        :type index: int
        :return: the type
        :rtype: str
        """
        return self.index_to_type.get(index, None)
