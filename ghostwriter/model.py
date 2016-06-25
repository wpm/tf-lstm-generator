import math

import numpy
import tensorflow as tf

from ghostwriter import logger
from ghostwriter.data import sequence_data_epoch, labeled_data


def train_language_model(tokens, hidden_size, rnn_depth, batch_size, time_steps, max_gradient, vocabulary_size):
    with tf.Graph().as_default(), tf.variable_scope("model", initializer=tf.random_uniform_initializer(-0.01, 0.01)):
        x = tf.placeholder(tf.int32, shape=[batch_size, time_steps], name="x")
        y = tf.placeholder(tf.int32, shape=[batch_size, time_steps], name="y")

        rnn_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        rnn_layers = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * rnn_depth)
        state = rnn_cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", shape=[vocabulary_size, hidden_size])
        token_embeddings = tf.nn.embedding_lookup(embedding, x, name="token_embeddings")

        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, time_steps, token_embeddings)]
        y_predicted, state = tf.nn.rnn(rnn_layers, inputs, initial_state=state)
        y_predicted = tf.reshape(tf.concat(1, y_predicted), [-1, hidden_size])

        w = tf.get_variable("w", [hidden_size, vocabulary_size])
        b = tf.get_variable("b", [vocabulary_size])

        logits = tf.matmul(y_predicted, w) + b
        loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                      [tf.reshape(y, [-1])],
                                                      [tf.ones([batch_size * time_steps])],
                                                      name="loss")
        cost = tf.reduce_sum(loss) / batch_size
        gradients, _ = tf.clip_by_global_norm(tf.gradients(cost, tf.trainable_variables()), max_gradient,
                                              name="clip_gradients")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), name="train")

        tokens = numpy.array(list(tokens))
        with tf.Session() as session:
            tf.initialize_all_variables().run()
            for epoch, iteration, xs, ys in labeled_data(
                    lambda: sequence_data_epoch(tokens, batch_size, time_steps)):
                if epoch > 5:
                    break
                c, _ = session.run([cost, train], feed_dict={x: xs, y: ys})
                if iteration % 10 == 0:
                    logger.info("Iteration %d (Epoch %d): Loss %0.3f" % (iteration, epoch, c))


def create_embeddings(data, batch_size,
                      vocabulary_size, embedding_size,
                      summary_directory, report_interval, iterations):
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
                                         feed_dict={inputs: input_batch, labels: numpy.reshape(label_batch, (-1, 1))})
                if i % report_interval == 0:
                    logger.info("Iteration %d: loss %0.5f" % (i, l))
                    train_writer.add_summary(s, global_step=i)
                if i == iterations:
                    break
            train_writer.flush()
        logger.info("%0.3f epochs" % (iterations * batch_size / n))


def summary_writer(summary_directory, graph):
    class NullSummaryWriter(object):
        def add_summary(self, *args, **kwargs):
            pass

        def flush(self):
            pass

    if summary_directory is not None:
        return tf.train.SummaryWriter(summary_directory, graph)
    else:
        return NullSummaryWriter()
