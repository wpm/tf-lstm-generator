import logging

import tensorflow as tf

__version__ = "1.0.0"


def noise_contrastive_estimation():
    with tf.Graph().as_default():
        pass


logger = logging.getLogger(__name__)


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)
