from __future__ import print_function

import argparse
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Language model")
    parser.add_argument("--log", default="INFO", help="logging level")
    parser.add_argument("text_files", metavar="FILE", type=argparse.FileType(), nargs="+", help="a text file")
    parser.add_argument("--max-vocabulary", metavar="N", type=int, help="maximum vocabulary size")
    args = parser.parse_args()

    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)
