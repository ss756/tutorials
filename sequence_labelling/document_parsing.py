import tensorflow as tf
# tf.disable_v2_behavior()
import numpy as np
import os

def write_graph(name):
    logdir = os.path.join("log", name)
    file_writer = tf.summary.FileWriter(logdir)
    with tf.Session() as sess:
        file_writer.add_graph(sess.graph)


n_classes = 10
num_layers = 2
state_size = 100
keep_prob = 0.5
vocab_size = 1000
embedding_size = 50

embedding_vectors = np.random.rand(vocab_size, embedding_size).astype(np.float32)
print("The embedding vector is", embedding_vectors)
labels = tf.placeholder(tf.int32, [None, None], name='labels')
x = tf.placeholder(tf.int32, [None, None], name="x")
seqlen = tf.placeholder(tf.int32, [None], name="seqlen")
word_embedding = tf.Variable(
    initial_value=embedding_vectors,
    trainable=False,
    name="word-embeddings"
)
rnn_inputs = tf.nn.embedding_lookup(word_embedding, x)
write_graph("step_1")


# defining the Bi-LSTM cell


def lstm_cell_with_dropout(state_size, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=state_size)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell=cell,
        output_keep_prob=keep_prob,
        state_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32)
    return cell

# defining the Bi-LSTM layer

def blstm_layer_with_dropout(inputs, seqlen, state_size, keep_prob, scope):
    cell = lstm_cell_with_dropout(state_size, keep_prob)
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell,
        cell_bw=cell,
        inputs=inputs,
        sequence_length=seqlen,
        dtype=tf.float32,
        scope=scope)
    return tf.concat([output_fw, output_bw], axis=-1)


# adding layers to the graph


for i in range(num_layers):
    with tf.name_scope("BLSTM-{}".format(i)) as scope:
        rnn_inputs = blstm_layer_with_dropout(
            rnn_inputs, seqlen, state_size, keep_prob, scope)

write_graph("step_2")


max_length = tf.shape(x)[1]

with tf.name_scope('logits'):
    logit_inputs = tf.reshape(rnn_inputs, [-1, 2 * state_size])
    logits = tf.layers.dense(logit_inputs, n_classes)
    logits = tf.reshape(logits, [-1, max_length, n_classes])
predictions = tf.argmax(logits, axis=-1, name="predictions")

write_graph("step_3")

with tf.name_scope('loss'):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name="cross_entropy")
    seqlen_mask = tf.sequence_mask(
        lengths=seqlen,
        maxlen=max_length,
        name='sequence_mask')
    loss = tf.boolean_mask(loss, mask=seqlen_mask)
    loss = tf.reduce_mean(loss, name="mean_loss")

with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss)

write_graph("step_4")

# Input placeholders and embeddings








# making predictions


max_length = tf.shape(x)[1]

with tf.name_scope('logits'):
    logit_inputs = tf.reshape(rnn_inputs, [-1, 2 * state_size])
    logits = tf.layers.dense(logit_inputs, n_classes)
    logits = tf.reshape(logits, [-1, max_length, n_classes])
predictions = tf.argmax(logits, axis=-1, name="predictions")

write_graph("step_3")


labels = tf.placeholder(tf.int32, [None, None], name='labels')
print("the labels are ", labels )

with tf.name_scope('loss'):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name="cross_entropy")
    seqlen_mask = tf.sequence_mask(
        lengths=seqlen,
        maxlen=max_length,
        name='sequence_mask')
    loss = tf.boolean_mask(loss, mask=seqlen_mask)
    loss = tf.reduce_mean(loss, name="mean_loss")

with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss)

write_graph("step_4")


# monitoring accuracy

train_summ = tf.summary.scalar("cross_entropy", loss)
n_examples = 1000
min_sequence_length = 5
max_sequence_length = 40
batch_size = 100

sequence_lengths = np.random.randint(min_sequence_length, max_sequence_length, size=n_examples)
X = np.zeros([n_examples,max_sequence_length], dtype=np.int32)
train_labels = np.zeros([n_examples,max_sequence_length], dtype=np.int32)

for i,length in enumerate(sequence_lengths):
    X[i,0:length] = np.random.randint(vocab_size, size=length)
    train_labels[i,0:length] = np.random.randint(n_classes, size=length)


n_epochs = 20

def data_gen():
    i = 0
    idx = 0
    while idx < len(sequence_lengths):
        slc = slice(idx, idx+batch_size)
        yield (X[slc], sequence_lengths[slc], train_labels[slc])
        i += 1
        idx = i * batch_size

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(os.path.join('log', 'train'), sess.graph)
    step = 0
    for i in range(n_epochs):
        for (X_batch, lengths, labels_batch) in data_gen():
            feed_dict = {x: X_batch, seqlen: lengths, labels: labels_batch}
            _, summ = sess.run([train_step, train_summ], feed_dict=feed_dict)
            train_writer.add_summary(summ, step)
            step += 1
    train_writer.close()