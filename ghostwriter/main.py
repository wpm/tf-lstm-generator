from __future__ import print_function

import argparse

from ghostwriter import configure_logger
from ghostwriter.tokenize import IndexedTokenizer, MultiFileLineEnumerator, EnglishWordTokenizer, CharacterTokenizer


def main():
    parser = argparse.ArgumentParser(description="Language model")
    parser.add_argument("--log", default="INFO", help="logging level")
    parser.add_argument("text_files", metavar="FILE", type=argparse.FileType(), nargs="+", help="a text file")
    parser.add_argument("--tokenizer", choices=["word", "character"], default="word", help="text tokenizer type")
    parser.add_argument("--max-vocabulary", metavar="N", type=int, help="maximum vocabulary size")
    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")

    lines = MultiFileLineEnumerator(args.text_files)
    if args.tokenizer == "word":
        tokens = EnglishWordTokenizer(lines)
    else:
        tokens = CharacterTokenizer(lines)
    indexes = IndexedTokenizer(tokens, args.max_vocabulary)
    for index in indexes:
        print("%d\t%s" % (index, indexes[index]))
