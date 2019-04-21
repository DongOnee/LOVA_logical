import argparse
from models import *
from utils import *
import tensorflow as tf
import time

###############
# argument
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--step", dest="step", type=int, metavar='<int>', default=5,
                    help="saver global step number (default=5)")
parser.add_argument("-e", "--epochs", dest="epochs", type=int, metavar='<int>', default=5,
                    help="Number of epochs (default=5)")
parser.add_argument("-ds", "--data-set", dest="dataset_count", type=int, metavar='<int>', default=20,
                    help="Number of data set (default=20)")
args = parser.parse_args()

###############
# Hyper params
global_step = args.step
lstm_size = [1024, 256, 1]
epochs = args.epochs
learning_rate = 0.4
batch_size_ = 100

print('#' * 5, "Global Step     :", global_step)
print('#' * 5, "LSTM Cell Size  :", lstm_size)
print('#' * 5, "Epochs          :", epochs)
print('#' * 5, "Learning Rate   :", learning_rate)
print('#' * 5, "Batch Size      :", batch_size_)

print("graph on")
with tf.device("/gpu:0"):
    with tf.Graph().as_default():
        # modeling
        essays, lengths, indice, scores, keep_prob = model_inputs()
        outputs, cell, init_state, final_states = build_lstm_layers(essays, lengths, lstm_size, keep_prob)
        predictions, losses, optimizer = build_cost_fn_and_opt(outputs, indice, scores, learning_rate)

        # to Tensorboard, saver
        loss_hist = tf.summary.scalar('loss_hist', losses)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Tensorboard Writer
            start_time = int(time.time())
            train_writer = tf.summary.FileWriter('board/train-'+str(start_time), sess.graph)
            test_writer = tf.summary.FileWriter('board/valid-'+str(start_time), sess.graph)
            sess.run(tf.global_variables_initializer())

            # train
            state = sess.run(init_state)
            for e in range(epochs):
                now_time = -time.time()
                for _index, (essays_, lengths_, scores_) in enumerate(get_batches5(batch_size=batch_size_), 1):
                    essay_indice = [[index, length - 1] for index, length in enumerate(lengths_)]
                    feed = {
                        essays:     essays_,
                        lengths:    lengths_,
                        indice:     essay_indice,
                        scores:     [[score] for score in scores_],
                        init_state: state,
                        keep_prob:  0.5
                    }
                    loss_, state, _ = sess.run([loss_hist, final_states, optimizer], feed_dict=feed)
                    if _index % 20 == 0:
                        train_writer.add_summary(loss_)

                now_time += time.time()
                now_time = time.gmtime(now_time)
                print("Epoch: {}/{}...\n".format(e + 1, epochs),
                      "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))

            # test
            now_time = -time.time()
            for _index, (essays_, lengths_, scores_) in enumerate(get_batches5(train_or_valid="valid", batch_size=batch_size_), 1):
                essay_indice = [[index, length - 1] for index, length in enumerate(lengths_)]
                feed = {
                    essays:     essays_,
                    lengths:    lengths_,
                    indice:     essay_indice,
                    scores:     [[score] for score in scores_],
                    init_state: state,
                    keep_prob:  1
                }
                loss_ = sess.run(loss_hist, feed_dict=feed)
                test_writer.add_summary(loss_)
            now_time += time.time()
            now_time = time.gmtime(now_time)
            print("Test Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))

            saver.save(sess, "logic_models/" + str(start_time))

            train_writer.close()
            test_writer.close()
