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
    :parm "sentences"   : sentences [batchsize, max_len, ???]
    :parm "lstm_sizes"  : stacked lstm hidden layer size
    :parm "keep_prob_"  : drop out value
    """
    fw_cells = [tf.contrib.rnn.LSTMCell(layer, name='basic_lstm_cell') for layer in hidden_layer]
    fw_drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in fw_cells]
    fw_stacked_cell = tf.contrib.rnn.MultiRNNCell(fw_drops)
    fw_init_state = fw_stacked_cell.zero_state(tf.shape(sentences)[0], tf.float32)

    bw_cells = [tf.contrib.rnn.LSTMCell(layer, name='basic_lstm_cell') for layer in hidden_layer]
    bw_drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in bw_cells]
    bw_stacked_cell = tf.contrib.rnn.MultiRNNCell(bw_drops)
    bw_init_state = bw_stacked_cell.zero_state(tf.shape(sentences)[0], tf.float32)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_stacked_cell, bw_stacked_cell, sentences,
                                                      sequence_length=sentences_length,
                                                      initial_state_fw=fw_init_state,
                                                      initial_state_bw=bw_init_state)

    return outputs


def build_cost_fn_and_opt(lstm_outputs, indice,  scores_, learning_rate, n_hidden):
    """
    Create the Loss function and Optimizer
    :parm "lstm_outputs"    : output of lstm layers
    :parm "embed_len"       : length of output of lstm
    :parm "scores_"         : true score value
    :parm "learning_rate"   : learning rate
    """
    outputs_fw = tf.gather_nd(lstm_outputs[0], indice)
    outputs_bw = tf.gather_nd(lstm_outputs[1], indice)
    outputs_concat = tf.concat([outputs_fw, outputs_bw], axis=1)
    weights = tf.Variable(tf.random_normal([n_hidden * 2], seed=10))
    bias = tf.Variable(tf.random_normal([1], seed=10))
    predictions = tf.matmul(outputs_concat, weights) + bias

    loss = tf.losses.mean_squared_error(scores_, predictions)
    # loss = tf.reduce_sum(tf.square(predictions - scores_))
    optimzer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return predictions, loss, optimzer

