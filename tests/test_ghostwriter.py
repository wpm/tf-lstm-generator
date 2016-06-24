from itertools import islice
from unittest import TestCase

import numpy
from numpy import array

from ghostwriter.data import skip_gram, LabeledData, batch_iterate
from ghostwriter.tokenize import CharacterTokenizer, EnglishWordTokenizer, MultiFileLineEnumerator, IndexedTokenizer, \
    IndexedVocabulary


class TestIndexing(TestCase):
    def test_full_vocabulary(self):
        v = IndexedVocabulary("the quick brown fox jumped over the lazy dog".split())
        self.assertEqual(len(v), 9)
        self.assertEqual(set(v.type_to_index.keys()), {"the", "quick", "brown", "fox", "jumped", "over", "lazy", "dog"})

    def test_limited_vocabulary(self):
        v = IndexedVocabulary("to be or not to be".split(), 2)
        self.assertEqual(len(v), 3)
        self.assertEqual(set(v.type_to_index.keys()), {"to", "be"})


class TestTokenization(TestCase):
    def test_character_tokenizer(self):
        t = CharacterTokenizer(["red blue", "green"])
        self.assertEqual(list(t), ["r", "e", "d", " ", "b", "l", "u", "e", "g", "r", "e", "e", "n"])

    def test_word_tokenizer(self):
        t = EnglishWordTokenizer(["The quick brown fox", "jumped over the lazy dog."])
        self.assertEqual(list(t), ["The", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog", "."])

    def test_multifile_line_enumerator(self):
        with open("file1.txt") as f1:
            with open("file2.txt") as f2:
                e = MultiFileLineEnumerator([f1, f2])
                self.assertEqual(list(e),
                                 ["The quick brown fox jumped over the lazy dog.\n",
                                  "As I walked out one evening,\n",
                                  "   Walking down Bristol Street,\n"])
                # File pointers are reset for each new iteration.
                self.assertEqual(list(e),
                                 ["The quick brown fox jumped over the lazy dog.\n",
                                  "As I walked out one evening,\n",
                                  "   Walking down Bristol Street,\n"])

    def test_indexed_tokenizer(self):
        t = IndexedTokenizer("to be or not to be".split())
        indexes = list(t)
        self.assertEqual(len(indexes), 6)
        self.assertEqual(t.vocabulary_size(), 5)
        tokens = [t[index] for index in indexes]
        self.assertEqual(tokens, ["to", "be", "or", "not", "to", "be"])

    def test_indexed_tokenizer_with_limited_vocabulary(self):
        t = IndexedTokenizer("to be or not to be".split(), 2)
        indexes = list(t)
        self.assertEqual(len(indexes), 6)
        self.assertEqual(t.vocabulary_size(), 3)
        tokens = [t[index] for index in indexes]
        self.assertEqual(tokens, ["to", "be", None, None, "to", "be"])


class TestWordContext(TestCase):
    def test_skip_gram(self):
        skip_grams = list(skip_gram("to be or not to be".split(), 1))
        self.assertEqual(skip_grams,
                         [("be", "to"), ("be", "or"),
                          ("or", "be"), ("or", "not"),
                          ("not", "or"), ("not", "to"),
                          ("to", "not"), ("to", "be")])


class TestData(TestCase):
    def test_labeled_data(self):
        vectors = [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
            [100, 110, 120],
            [130, 140, 150]
        ]
        labels = [1, 2, 3, 4, 5]
        data = LabeledData(vectors, labels)
        # 4 batches of size 2 loops around to the first item.
        batches = list(islice(data.batches(2), 4))
        expected = [
            (array([[10, 20, 30], [40, 50, 60]]), array([1, 2])),
            (array([[70, 80, 90], [100, 110, 120]]), array([3, 4])),
            (array([[130, 140, 150], [10, 20, 30]]), array([5, 1])),
            (array([[40, 50, 60], [70, 80, 90]]), array([2, 3]))
        ]
        numpy.testing.assert_equal(batches, expected)

    def test_batch_iterate(self):
        x = list(islice(batch_iterate("abcd", 2), 5))
        self.assertEqual(x, ["ab", "cd", "ab", "cd", "ab"])
        x = list(islice(batch_iterate("abcd", 3), 5))
        self.assertEqual(x, ["abc", "dab", "cda", "bcd", "abc"])
        x = list(islice(batch_iterate("abcd", 4), 3))
        self.assertEqual(x, ["abcd", "abcd", "abcd"])
        x = list(islice(batch_iterate("abcd", 1000), 3))
        self.assertEqual(x, ["abcd", "abcd", "abcd"])
