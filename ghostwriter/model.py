import math

import tensorflow as tf
from numpy import reshape

from ghostwriter import logger


def noise_contrastive_estimation(data, batch_size,
                                 vocabulary_size, embedding_size,
                                 summary_directory, report_interval,
                                 iterations):
    n = len(data)
    logger.info("%d training items, %d iterations per epoch" % (n, math.ceil(n / batch_size)))
    shape = [vocabulary_size, embedding_size]

    with tf.Graph().as_default():
        with tf.name_scope("Input"):
            inputs = tf.placeholder(tf.int32, shape=[batch_size], name="inputs")
            labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name="labels")

        with tf.name_scope("Embedding"):
            embeddings = tf.Variable(tf.random_uniform(shape, -1.0, 1.0), name="embeddings")
            lookup = tf.nn.embedding_lookup(embeddings, inputs, name="lookup")
            w = tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(embedding_size)), name="weights")
            b = tf.Variable(tf.zeros(vocabulary_size), name="bias")

        with tf.name_scope("Training"):
            iteration = tf.Variable(0, name="iteration", trainable=False)
            loss = tf.reduce_mean(tf.nn.nce_loss(w, b, lookup, labels, 5, vocabulary_size), name="loss")
            tf.scalar_summary(loss.op.name, loss)
            training_step = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss, global_step=iteration)

        summary = tf.merge_all_summaries()

        with tf.Session() as session:
            train_writer = summary_writer(summary_directory, session.graph)
            session.run(tf.initialize_all_variables())
            for input_batch, label_batch in data.batches(batch_size):
                s, i, l, _ = session.run([summary, iteration, loss, training_step],
                                         feed_dict={inputs: input_batch, labels: reshape(label_batch, (-1, 1))})
                if i % report_interval == 0:
                    logger.info("Iteration %d: loss %0.5f" % (i, l))
                    train_writer.add_summary(s, global_step=i)
                if i == iterations:
                    break
            train_writer.flush()
        logger.info("%0.3f epochs" % (iterations * batch_size / n))


def summary_writer(summary_directory, graph):
    class NullSummaryWriter(object):
        def add_summary(self, *args):
            pass

        def flush(self):
            pass

    if summary_directory is not None:
        return tf.train.SummaryWriter(summary_directory, graph)
    else:
        return NullSummaryWriter()
