import argparse
import tensorflow as tf
import numpy as np
from utils import *
from models import *
import time

parser = argparse.ArgumentParser()
parser.add_argument("-e","--epochs", dest="epochs", type=int, metavar='<int>', default=10, help="Number of epochs (default=10)")
parser.add_argument("-s","--step", dest="step", type=int, metavar='<int>', default=3, help="saver global step number (default=3)")
args = parser.parse_args()

##### Hyper params
global_step = args.step
epochs = args.epochs
lstm_sizes = [1024, 256]
learning_rate = 0.002
batch_size = 10
limite = -1

print('#' * 5, "global_step     :", global_step)
print('#' * 5, "epochs          :", epochs)
print('#' * 5, "learning_rate   :", learning_rate)
print('#' * 5, "lstm_sizes      :", lstm_sizes)

##### data preprocessing
train_x, train_y, valid_x, valid_y = get_data(limite, limite)

with tf.Graph().as_default():
    ##### modeling
    essay, score, essay_lens, keep_prob_ = model_inputs()
    embedded = build_embedding_layer(essay, batch_size)
    lstm_outputs, lstm_cell, initial_state, final_state = build_lstm_layers(lstm_sizes, embedded, essay_lens, keep_prob_, batch_size)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, essay_lens, score, learning_rate)
    accuracy = build_accuracy(predictions, score)

    saver = tf.train.Saver()

    ##### Training
    with tf.Session() as sess:
        n_batches = len(train_x)//batch_size
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            now_time = time.time()
            state = sess.run(initial_state)
            train_acc = []

            for ii, (x, y, ll) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                feed = {essay: x,
                        score: y,
                        keep_prob_: 0.5,
                        essay_lens: ll,
                        initial_state: state}
                loss_, state, _ = sess.run([loss, final_state, optimizer], feed_dict=feed)
                train_acc.append(loss_)

                ##### Validation
                if (ii + 1) % n_batches == 0:
                    val_acc = []
                    val_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
                    for xx, yy, lll in get_batches(train_x, train_y, batch_size):
                        feed = {essay: xx,
                                score: yy,
                                keep_prob_: 1,
                                essay_lens: lll,
                                initial_state: val_state}
                        val_loss, val_state = sess.run([loss, final_state], feed_dict=feed)
                        val_acc.append(val_loss)
                    now_time -= time.time()
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec),
                          "Batch: {}/{}...".format(ii+1, n_batches),
                          "Train Loss: {:.3f}...".format(loss_),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Loss: {:.3f}".format(val_loss))
            saver.save(sess, "logic_models/dongs", global_step=global_step)
