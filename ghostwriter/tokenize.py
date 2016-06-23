from itertools import chain

from nltk import word_tokenize


class MultiFileLineEnumerator(object):
    """
    Enumerate over all the lines in a sequence of files.
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
        return chain(*(text_file.readlines() for text_file in self.text_files))

    def reset(self):
        """
        Reset all file pointers to the beginning of the file
        """
        for text_file in self.text_files:
            text_file.seek(0)


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
