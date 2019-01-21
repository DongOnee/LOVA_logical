import tensorflow as tf
import tensorflow_hub as hub
import embedding2

train_data_path = '../data/training_set_rel3.tsv'
valid_data_path = '../data/valid_set.tsv'
valid_predic_path = '../data/valid_sample_submission_2_column.csv'

train_x, train_y, valid_x, valid_y = embedding2.get_data([train_data_path,
                                                          valid_data_path,
                                                          valid_predic_path])

from nltk.tokenize import sent_tokenize, word_tokenize

train_x = [ sent_tokenize(pa) for pa in train_x]
train_y = [ [pa] for pa in train_y]
valid_x = [ sent_tokenize(pa) for pa in valid_x]
valid_y = [ [pa] for pa in valid_y]

def model_inputs():
    """
    Create the model inputs
    """
    inputs_ = tf.placeholder(tf.string, [None], name='inputs') # sentences
    scores_ = tf.placeholder(tf.int32, [1], name='scores')
    keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs_, scores_, keep_prob_

def build_embedding_layer(input_paragraph):
    """
    Create the embedding layer
    """
    module_url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(module_url, trainable=True)
    embeddings = embed(input_paragraph,
                       signature="default",
                       as_dict=True)
    
    return embeddings['elmo']

def build_lstm_layers(lstm_sizes, embed, keep_prob_):
    """
    Create the LSTM layers
    """
    lstms = [tf.contrib.rnn.LSTMCell(size, name='basic_lstm_cell') for size in lstm_sizes]
    # Add dropout to the cell
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    # Getting an initial state of all zeros
#     initial_state = cell.zero_state(batch_size, tf.float32)
    
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    
    return lstm_outputs, cell, final_state

def build_cost_fn_and_opt(lstm_outputs, scores_, learning_rate):
    """
    Create the Loss function and Optimizer
    """
    predictions = tf.contrib.layers.fully_connected(lstm_outputs[:, -1], 1, activation_fn=tf.sigmoid)
#     tensor1 = tf.constant(1.0, shape=[1, 256])
#     tensor2 = tf.matmul(tensor1, predictions)
    loss = tf.losses.mean_squared_error(scores_, tf.reshape(predictions, [-1]))
    optimzer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    
    return predictions, loss, optimzer

def build_accuracy(predictions, scores_):
    """
    Create accuracy
    """
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), scores_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return accuracy

def build_and_train_network(lstm_sizes, epochs, learning_rate, keep_prob, 
                            train_x, val_x, train_y, val_y):
    
    i, s, kp = model_inputs()
    em = build_embedding_layer(i)
    lstout, cell, finstd = build_lstm_layers(lstm_sizes, em, kp)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstout, s, learning_rate)
    accuracy = build_accuracy(predictions, s)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
#             state = sess.run(initial_state)
            
            train_acc = []
            for ii, (x, y) in enumerate(zip(train_x, train_y)):
                feed = {i: x,
                        s: y,
                        kp: keep_prob}
                loss_, state, _,  batch_acc = sess.run([loss, finstd, optimizer, accuracy], feed_dict=feed)
                train_acc.append(batch_acc)

                if (ii + 1) % 5 == 0:
                    
                    val_acc = []
#                     val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                    for xx, yy in zip(val_x, val_y):
                        feed = {i: xx,
                                s: yy,
                                kp: 1}
                        val_batch_acc, val_state = sess.run([accuracy, finstd], feed_dict=feed)
                        val_acc.append(val_batch_acc)
                    
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Train Loss: {:.3f}...".format(loss_),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Accuracy: {:.3f}".format(np.mean(val_acc)))
    
        saver.save(sess, "checkpoints/sentiment.ckpt")
        
lstm_sizes = [1024, 256, 64]
keep_prob = 0.5
epochs=10
lr = 0.01
import numpy as np

with tf.Graph().as_default():
    build_and_train_network(lstm_sizes, epochs, lr, keep_prob,
                            np.array(train_x), np.array(valid_x),
                            np.array(train_y), np.array(valid_y))


