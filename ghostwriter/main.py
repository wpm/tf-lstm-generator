"""Analyze and generate text."""

from __future__ import print_function

import argparse
import textwrap
from argparse import RawDescriptionHelpFormatter as Raw

from ghostwriter import configure_logger, __version__, logger
from ghostwriter.data import skip_gram_data_set
from ghostwriter.model import noise_contrastive_estimation
from ghostwriter.tokenize import IndexedTokenizer, MultiFileLineEnumerator, EnglishWordTokenizer, CharacterTokenizer


def main():
    parser = argparse.ArgumentParser(description="Ghostwriter version %s" % __version__)

    shared_arguments = argparse.ArgumentParser(add_help=False)
    shared_arguments.add_argument("--log", default="INFO", help="logging level")
    shared_arguments.add_argument("text_files", type=argparse.FileType(), nargs="+", help="a text file")
    shared_arguments.add_argument("--tokenizer", choices=["word", "character"], default="word",
                                  help="text tokenizer type, default word")
    shared_arguments.add_argument("--min-frequency", type=int,
                                  help="minimum token frequency for inclusion in the vocabulary")
    shared_arguments.add_argument("--max-vocabulary", type=int, help="maximum vocabulary size")

    subparsers = parser.add_subparsers(title="Machine-assisted writing", description=__doc__)

    train = subparsers.add_parser("train", parents=[shared_arguments], help="train a language model",
                                  formatter_class=Raw,
                                  description=textwrap.dedent("""
                                  Train a language model with noise contrastive estimation.

                                  See ghostwriter tokenize --help for details about tokenization."""))
    train.add_argument("--width", type=int, default=1, help="skip-gram window size, default 1")
    train.add_argument("--batch-size", type=int, default=100, help="minibatch size, default 100")
    train.add_argument("--embedding-size", type=int, default=128, help="vocabulary embedding size, default 128")
    train.add_argument("--summary-directory", help="directory into which to write summary files")
    train.add_argument("--report-interval", type=int, default=100,
                       help="log and write summary after this many iterations, default 100")
    train.add_argument("--iterations", type=int, default=1000, help="total training iterations, default 1000")
    train.set_defaults(func=train_command)

    tokenize = subparsers.add_parser("tokenize", parents=[shared_arguments], help="tokenize text files",
                                     formatter_class=Raw,
                                     description=textwrap.dedent("""
        Tokenize a set of files into indexed lists of either words or characters. Print
        the indexes and their types on separate lines.

        Types with a token frequency below a minimum value may be omitted from the
        vocabulary, and the vocabulary size may be capped, keeping the most common
        types. If a type is omitted from the vocabulary, it is mapped to a special
        out of vocabulary type."""))
    tokenize.set_defaults(func=tokenize_command)

    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


def tokenize_command(args):
    tokens = indexed_tokens(args.text_files, args.tokenizer, args.min_frequency, args.max_vocabulary)
    n = 0
    for index in tokens:
        # noinspection PyStringFormat,PyTypeChecker
        print("%d\t%s" % (index, tokens[index]))
        n += 1
    logger.info("%d tokens, %d types" % (n, tokens.vocabulary_size()))


def train_command(args):
    tokens = indexed_tokens(args.text_files, args.tokenizer, args.min_frequency, args.max_vocabulary)
    data = skip_gram_data_set(tokens, args.width)
    noise_contrastive_estimation(data, args.batch_size,
                                 tokens.vocabulary_size(), args.embedding_size,
                                 args.summary_directory, args.report_interval,
                                 args.iterations)


def indexed_tokens(text_files, tokenizer, min_frequency, max_vocabulary):
    lines = MultiFileLineEnumerator(text_files)
    if tokenizer == "word":
        tokens = EnglishWordTokenizer(lines)
    else:
        tokens = CharacterTokenizer(lines)
    return IndexedTokenizer(tokens, min_frequency, max_vocabulary)
