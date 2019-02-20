import tensorflow as tf
import tensorflow_hub as hub


def model_inputs():
    """
    Create the model inputs
    """
    inputs_ = tf.placeholder(tf.string, [None, None], name='inputs') # paragraphs
    scores_ = tf.placeholder(tf.float32, [None, 1], name='scores')
    lens_ = tf.placeholder(tf.int32, [None], name='lengths')
    keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')

    return inputs_, scores_, lens_, keep_prob_


def build_embedding_layer(input_paragraphs, batch_size):
    """
    Create the embedding layer (Elmo)
    :parm "input_paragraphs" : list of paragraphs
    :parm "batch_size" : data batch size
    """
    elmo_module_url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(elmo_module_url, trainable=True)
    paragraphs = tf.split(input_paragraphs, num_or_size_splits=batch_size, axis=0)
    paragraph_tensor_list = []
    for paragraph in paragraphs:
        paragraph = tf.reshape(paragraph, [100]) # [1, 100] => [100]
        embeds = embed(paragraph, signature="default", as_dict=True)['elmo']
        rm = tf.reduce_mean(embeds, 1) # [100, 1024]
        paragraph_tensor_list.append(rm)

    return tf.stack(paragraph_tensor_list, 0) # tensor list => upper rank tensor


def build_lstm_layers(lstm_sizes, embed, embed_len, keep_prob_, batch_size):
    """
    Create the LSTM layers
    :parm "lstm_sizes"  : stacked lstm hidden layer size
    :parm "embed"       : embedded sentences vector representation
    :parm "keep_prob_"  : drop out value
    """
    lstms = [tf.contrib.rnn.LSTMCell(size, name='basic_lstm_cell') for size in lstm_sizes]
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    initial_state = cell.zero_state(batch_size, tf.float32)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=embed_len, initial_state=initial_state)

    return lstm_outputs, cell, initial_state, final_state


def build_cost_fn_and_opt(lstm_outputs, embed_len, scores_, learning_rate):
    """
    Create the Loss function and Optimizer
    :parm "lstm_outputs"    : output of lstm layers
    :parm "embed_len"       : length of output of lstm
    :parm "scores_"         : true score value
    :parm "learning_rate"   : learning rate
    """
    predic_input = tf.unstack(lstm_outputs);
    length_list = tf.unstack(embed_len, 10);
    predics = []
    for t, i in zip(predic_input, length_list):
        ret = tf.split(t, [i, 100-i], 0)[0][-1]
        predics.append(ret)
    predictions = tf.contrib.layers.fully_connected(tf.stack(predics, 0), 1, activation_fn=tf.sigmoid)
    predictions = tf.identity(predictions, name="predictions")
    loss = tf.losses.mean_squared_error(scores_, predictions)
    optimzer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

    return predictions, loss, optimzer

# 필요하나... ?
def build_accuracy(predictions, scores_):
    """
    Create accuracy
    :parm "predictions" : model output
    :parm "scores_"     : true score
    """
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.float32), scores_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy
