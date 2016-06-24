"""Analyze and generate text."""

from __future__ import print_function

import argparse
import textwrap
from argparse import RawDescriptionHelpFormatter as Raw

from ghostwriter import configure_logger, __version__
from ghostwriter.data import skip_gram_data_set
from ghostwriter.model import noise_contrastive_estimation
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

    train = subparsers.add_parser("train", parents=[shared_arguments], help="train a language model",
                                  description="""Train a language model with noise constrastive estimation.""")
    train.add_argument("--width", type=int, default=1, help="skip-gram window size")
    train.add_argument("--batch-size", type=int, default=100, help="minibatch size")
    train.add_argument("--embedding-size", type=int, default=128, help="embedding vector size")
    train.add_argument("--summary-directory", help="directory into which to write summary files")
    train.add_argument("--report-interval", type=int, default=100,
                       help="write summary and log after this many iterations")
    train.add_argument("--iterations", type=int, default=1000, help="total training iterations")
    train.set_defaults(func=train_command)

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
    tokens = indexed_tokens(args.text_files, args.tokenizer, args.max_vocabulary)
    for index in tokens:
        # noinspection PyStringFormat,PyTypeChecker
        print("%d\t%s" % (index, tokens[index]))


def train_command(args):
    tokens = indexed_tokens(args.text_files, args.tokenizer, args.max_vocabulary)
    data = skip_gram_data_set(tokens, args.width)
    noise_contrastive_estimation(data, args.batch_size,
                                 tokens.vocabulary_size(), args.embedding_size,
                                 args.summary_directory, args.report_interval,
                                 args.iterations)


def indexed_tokens(text_files, tokenizer, max_vocabulary):
    lines = MultiFileLineEnumerator(text_files)
    if tokenizer == "word":
        tokens = EnglishWordTokenizer(lines)
    else:
        tokens = CharacterTokenizer(lines)
    return IndexedTokenizer(tokens, max_vocabulary)
