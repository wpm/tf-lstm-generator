from __future__ import print_function

import textwrap

"""Analyze and generate text."""

import argparse
from argparse import RawDescriptionHelpFormatter as Raw

from ghostwriter import configure_logger, __version__
from ghostwriter.tokenize import IndexedTokenizer, MultiFileLineEnumerator, EnglishWordTokenizer, CharacterTokenizer


def main():
    parser = argparse.ArgumentParser(description="Ghostwriter version %s" % __version__)

    shared_arguments = argparse.ArgumentParser(add_help=False)
    shared_arguments.add_argument("--log", metavar="LEVEL", default="INFO", help="logging level")
    shared_arguments.add_argument("text_files", metavar="FILE", type=argparse.FileType(), nargs="+", help="a text file")
    shared_arguments.add_argument("--tokenizer", choices=["word", "character"], default="word",
                                  help="text tokenizer type")
    shared_arguments.add_argument("--max-vocabulary", metavar="N", type=int, help="maximum vocabulary size")

    subparsers = parser.add_subparsers(title="Machine-assisted writing", description=__doc__)
    tokenize = subparsers.add_parser("tokenize", parents=[shared_arguments], help="tokenize text files",
                                     formatter_class=Raw,
                                     description=textwrap.dedent("""
    Tokenize a set of files into indexed lists of either words or characters.
    Print the indexes and the types on separate lines.
    If a maximum vocabulary size is specified, only the most common types will be kept.
    The others will be mapped to an out of vocabulary type."""))
    tokenize.set_defaults(func=tokenize_command)

    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


def tokenize_command(args):
    lines = MultiFileLineEnumerator(args.text_files)
    if args.tokenizer == "word":
        tokens = EnglishWordTokenizer(lines)
    else:
        tokens = CharacterTokenizer(lines)
    indexes = IndexedTokenizer(tokens, args.max_vocabulary)
    for index in indexes:
        # noinspection PyStringFormat,PyTypeChecker
        print("%d\t%s" % (index, indexes[index]))
