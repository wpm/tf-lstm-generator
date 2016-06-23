from itertools import chain

from nltk import word_tokenize

from ghostwriter.index import IndexedVocabulary


class MultiFileLineEnumerator(object):
    """
    Enumerate over all the lines in a sequence of files.

    File pointers are reset for new new iteration, so iteration is idempotent.
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
