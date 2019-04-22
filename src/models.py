import tensorflow as tf

########################
# Model 1
#


def model_inputs():
    """
    Create the model inputs
    """
    inputs_ = tf.placeholder(tf.float32, [None, None, None], name='essays')
    scores_ = tf.placeholder(tf.float32, [None, 1], name='scores')
    lens_ = tf.placeholder(tf.int32, [None], name='essay_lengths')
    indice_ = tf.placeholder(tf.int32, [None, 2], name='indice')
    keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')

    return inputs_, lens_, indice_, scores_, keep_prob_


def build_lstm_layers(sentences, sentences_length, hidden_layer, keep_prob_):
    """
    Create the LSTM layers
    :parm "lstm_sizes"  : stacked lstm hidden layer size
    :parm "embed"       : embedded sentences vector representation
    :parm "keep_prob_"  : drop out value
    """
    embed = tf.reshape(sentences, [-1, 100, 1024])
    lstms = [tf.contrib.rnn.LSTMCell(layer, name='basic_lstm_cell') for layer in hidden_layer]
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    init_state = cell.zero_state(100, tf.float32)

    outputs, states = tf.nn.dynamic_rnn(cell, embed, initial_state=init_state, sequence_length=sentences_length)

    return outputs, cell, init_state, states


def build_cost_fn_and_opt(lstm_outputs, indice,  scores_, learning_rate):
    """
    Create the Loss function and Optimizer
    :parm "lstm_outputs"    : output of lstm layers
    :parm "embed_len"       : length of output of lstm
    :parm "scores_"         : true score value
    :parm "learning_rate"   : learning rate
    """
    # last_sentences
    predictions = tf.gather_nd(lstm_outputs, indice)  # [batchsize, cell.outputsize]
    # predictions = tf.contrib.layers.fully_connected(last_sentences, 1, activation_fn=tf.sigmoid)

    loss = tf.losses.sum_squared_error(scores_, predictions)
    # loss = tf.reduce_sum(tf.square(predictions - scores_))
    optimzer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return predictions, loss, optimzer

