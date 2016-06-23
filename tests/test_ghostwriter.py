from unittest import TestCase

import ghostwriter.tokenize


class TestTokenizer(TestCase):
    def test_character_tokenizer(self):
        t = ghostwriter.tokenize.CharacterTokenizer(["red blue", "green"])
        self.assertEqual(list(t), ["r", "e", "d", " ", "b", "l", "u", "e", "g", "r", "e", "e", "n"])

    def test_word_tokenizer(self):
        t = ghostwriter.tokenize.EnglishWordTokenizer(["The quick brown fox", "jumped over the lazy dog."])
        self.assertEqual(list(t), ["The", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog", "."])

    def test_multifile_line_enumerator(self):
        with open("file1.txt") as f1:
            with open("file2.txt") as f2:
                e = ghostwriter.tokenize.MultiFileLineEnumerator([f1, f2])
                self.assertEqual(list(e),
                                 ["The quick brown fox jumped over the lazy dog.\n",
                                  "As I walked out one evening,\n",
                                  "   Walking down Bristol Street,\n"])
