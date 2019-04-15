import tensorflow as tf




########################
# Model 1
#


def model_inputs():
    """
    Create the model inputs
    """
    inputs_ = tf.placeholder(tf.float32, [None, None, None], name='essays')  # vectors
    scores_ = tf.placeholder(tf.float32, [None, 1], name='scores')
    lens_ = tf.placeholder(tf.int32, [None], name='essay_lengths')
    lens_pad_ = tf.placeholder(tf.int32, [None, 2], name='lengths_pad')
    batch_size_ = tf.placeholder(tf.int32, name='batch_size')
    keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')

    return inputs_, lens_, lens_pad_, scores_, batch_size_, keep_prob_


def build_lstm_layers(lstm_sizes, embed, embed_len, batch_size, keep_prob_):
    """
    Create the LSTM layers
    :parm "lstm_sizes"  : stacked lstm hidden layer size
    :parm "embed"       : embedded sentences vector representation
    :parm "keep_prob_"  : drop out value
    """
    embed = tf.reshape(embed, [batch_size, 100, 1024])
    lstms = [tf.contrib.rnn.LSTMCell(size, name='basic_lstm_cell') for size in lstm_sizes]
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
    cell = tf.contrib.rnn.MultiRNNCell(drops)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=embed_len, dtype=tf.float32)

    return lstm_outputs, cell, final_state


def build_cost_fn_and_opt(lstm_outputs, lengths,  scores_, learning_rate, batch_size):
    """
    Create the Loss function and Optimizer
    :parm "lstm_outputs"    : output of lstm layers
    :parm "embed_len"       : length of output of lstm
    :parm "scores_"         : true score value
    :parm "learning_rate"   : learning rate
    """
    predictions = tf.gather_nd(lstm_outputs, lengths)
    # lstm_outputs = tf.reduce_mean(lstm_outputs, 1)
    predictions = tf.contrib.layers.fully_connected(predictions, 1, activation_fn=tf.sigmoid)
    predictions = tf.reshape(predictions, [batch_size, 1], name="result")

    loss = tf.losses.mean_squared_error(scores_, predictions)
    optimzer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

    return predictions, loss, optimzer
