from unittest import TestCase

from ghostwriter.tokenize import CharacterTokenizer, EnglishWordTokenizer, MultiFileLineEnumerator, IndexedTokenizer, \
    IndexedVocabulary


class TestIndexing(TestCase):
    def test_full_vocabulary(self):
        v = IndexedVocabulary("the quick brown fox jumped over the lazy dog".split())
        self.assertEqual(len(v), 8)

    def test_limited_vocabulary(self):
        v = IndexedVocabulary("to be or not to be".split(), 2)
        self.assertEqual(len(v), 2)
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
        tokens = [t[index] for index in indexes]
        self.assertEqual(tokens, ["to", "be", "or", "not", "to", "be"])

    def test_indexed_tokenizer_with_limited_vocabulary(self):
        t = IndexedTokenizer("to be or not to be".split(), 2)
        indexes = list(t)
        self.assertEqual(len(indexes), 6)
        tokens = [t[index] for index in indexes]
        self.assertEqual(tokens, ["to", "be", None, None, "to", "be"])
