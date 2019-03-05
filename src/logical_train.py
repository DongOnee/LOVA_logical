import argparse
from models import *
from utils import *
from embedding import *
import tensorflow as tf
import time

###############
# argument
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", dest="epochs", type=int, metavar='<int>', default=10,
                    help="Number of epochs (default=10)")
parser.add_argument("-s", "--step", dest="step", type=int, metavar='<int>', default=3,
                    help="saver global step number (default=3)")
args = parser.parse_args()

###############
# Hyper params
global_step = args.step
epochs = args.epochs
lstm_sizes = [1024, 256]
batch_size = 5
lr = 0.1

print('#' * 5, "global_step     :", global_step)
print('#' * 5, "epochs          :", epochs)
print('#' * 5, "learning_rate   :", lr)
print('#' * 5, "lstm_sizes      :", lstm_sizes)

# data pre-processing
train_essay, train_score, valid_essay, valid_score = get_data()

vector_essay_train, lengths_essay_train = embedding_parag(train_essay)
train_score = train_score
vector_essay_valid, lengths_essay_valid = embedding_parag(valid_essay)
valid_score = valid_score

print("graph on")
with tf.Graph().as_default():
    # modeling
    i, l, l_p, s, bs, kp = model_inputs()
    lo, c, final = build_lstm_layers(lstm_sizes, i, l, bs, kp)
    pre, loss, opt = build_cost_fn_and_opt(lo, l_p, s, lr, bs)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        n_batches = len(vector_essay_train) // batch_size
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            now_time = -time.time()
            state = sess.run(c.zero_state(batch_size, tf.float32))
            train_acc = []

            for ii, (x, ll, y) in enumerate(get_batches(vector_essay_train, lengths_essay_train, train_score, batch_size), 1):
                llp = [[sibal, ssibal - 1] for sibal, ssibal in enumerate(ll)]
                feed = {i: x,
                        l: ll,
                        l_p: llp,
                        s: y,
                        bs: batch_size,
                        kp: 0.5}
                loss_, state, _ = sess.run([loss, final, opt], feed_dict=feed)
                train_acc.append(loss_)

                ##### Validation
                if (ii + 1) % n_batches == 0:
                    val_acc = []
                    val_state = sess.run(c.zero_state(batch_size, tf.float32))
                    for xx, lll, yy in get_batches(vector_essay_valid, lengths_essay_valid, valid_score, batch_size):
                        lllp = [[sibal, ssibal - 1] for sibal, ssibal in enumerate(lll)]
                        feed = {i: xx,
                                l: lll,
                                l_p: lllp,
                                s: yy,
                                bs: batch_size,
                                kp: 1}
                        val_loss, val_state = sess.run([loss, final], feed_dict=feed)
                        val_acc.append(val_loss)
                    now_time += time.time()
                    now_time = time.gmtime(now_time)
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec),
                          "Batch: {}/{}...".format(ii + 1, n_batches),
                          "Train Loss: {:.3f}...".format(loss_),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Loss: {:.3f}".format(val_loss))
        saver.save(sess, "logic_models/dongs", global_step=global_step)


