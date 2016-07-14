"""Analyze and generate text."""

from __future__ import print_function

import argparse
import textwrap
from argparse import RawDescriptionHelpFormatter as Raw

from ghostwriter import configure_logger, __version__, logger
from ghostwriter.data import skip_gram_data_set
from ghostwriter.model import create_embeddings, train_language_model
from ghostwriter.tokenize import IndexedTokenizer, MultiFileLineEnumerator, EnglishWordTokenizer, CharacterTokenizer


def main():
    parser = argparse.ArgumentParser(description="Ghostwriter version %s" % __version__)

    tokenize_arguments = argparse.ArgumentParser(add_help=False)
    tokenize_arguments.add_argument("--log", default="INFO", help="logging level")
    tokenize_arguments.add_argument("text_files", type=argparse.FileType(), nargs="+", help="a text file")
    tokenize_arguments.add_argument("--tokenizer", choices=["word", "character"], default="word",
                                    help="text tokenizer type, default word")
    tokenize_arguments.add_argument("--min-frequency", type=int,
                                    help="minimum token frequency for inclusion in the vocabulary")
    tokenize_arguments.add_argument("--max-vocabulary", type=int, help="maximum vocabulary size")

    train_arguments = argparse.ArgumentParser(add_help=False)
    train_arguments.add_argument("--batch-size", type=int, default=100, help="batch size, default 100")
    train_arguments.add_argument("--summary-directory",
                                 help="directory into which to write summary files, default do not summarize")
    train_arguments.add_argument("--max-epoch", type=int, default=5, help="number of training epochs, default 5")
    train_arguments.add_argument("--max-iteration", type=int, help="number of training iterations, default unlimited")
    train_arguments.add_argument("--report-interval", type=int, default=10,
                                 help="log and write summary after this many iterations, default 10")

    subparsers = parser.add_subparsers(title="Machine-assisted writing", description=__doc__)

    train = subparsers.add_parser("train", parents=[tokenize_arguments, train_arguments], help="train a language model")
    train.add_argument("--hidden-size", type=int, default=100, help="hidden variables in RNN, default 100")
    train.add_argument("--rnn-depth", type=int, default=1, help="RNN depth, default 1")
    train.add_argument("--time-steps", type=int, default=10, help="RNN time steps to unroll, default 10")
    train.set_defaults(func=train_command)

    embed = subparsers.add_parser("embed", parents=[tokenize_arguments, train_arguments], help="create word embeddings",
                                  formatter_class=Raw,
                                  description=textwrap.dedent("""
                                  Create token embeddings with noise contrastive estimation.

                                  See ghostwriter tokenize --help for details about tokenization."""))
    embed.add_argument("--width", type=int, default=1, help="skip-gram window size, default 1")
    embed.add_argument("--embedding-size", type=int, default=128, help="vocabulary embedding size, default 128")
    embed.add_argument("--iterations", type=int, default=1000, help="total training iterations, default 1000")
    embed.set_defaults(func=embed_command)

    tokenize = subparsers.add_parser("tokenize", parents=[tokenize_arguments], help="tokenize text files",
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


def train_command(args):
    tokens = indexed_tokens(args.text_files, args.tokenizer, args.min_frequency, args.max_vocabulary)
    train_language_model(tokens,
                         args.hidden_size, args.rnn_depth,
                         args.batch_size, args.time_steps,
                         5,
                         tokens.vocabulary_size(),
                         args.report_interval, args.max_epoch, args.max_iteration,
                         args.summary_directory)


def embed_command(args):
    tokens = indexed_tokens(args.text_files, args.tokenizer, args.min_frequency, args.max_vocabulary)
    data = skip_gram_data_set(tokens, args.width)
    create_embeddings(data, args.batch_size,
                      tokens.vocabulary_size(), args.embedding_size,
                      args.summary_directory, args.report_interval,
                      args.iterations)


def tokenize_command(args):
    tokens = indexed_tokens(args.text_files, args.tokenizer, args.min_frequency, args.max_vocabulary)
    n = 0
    for index in tokens:
        # noinspection PyStringFormat,PyTypeChecker
        print("%d\t%s" % (index, tokens[index]))
        n += 1
    logger.info("%d tokens, %d types" % (n, tokens.vocabulary_size()))


def indexed_tokens(text_files, tokenizer, min_frequency, max_vocabulary):
    lines = MultiFileLineEnumerator(text_files)
    if tokenizer == "word":
        tokens = EnglishWordTokenizer(lines)
    else:
        tokens = CharacterTokenizer(lines)
    return IndexedTokenizer(tokens, min_frequency, max_vocabulary)
