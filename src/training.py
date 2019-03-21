import argparse
from models import *
from utils import *
from embedding import *
import tensorflow as tf
import time

###############
# argument
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--step", dest="step", type=int, metavar='<int>', default=4,
                    help="saver global step number (default=4)")
parser.add_argument("-e", "--epochs", dest="epochs", type=int, metavar='<int>', default=5,
                    help="Number of epochs (default=5)")
args = parser.parse_args()

###############
# Hyper params
globalStep = args.step
epochs = args.epochs
lstmSizes = [1024, 256]
batchSize = 5
lr = 0.1

print('#' * 5, "global_step     :", globalStep)
print('#' * 5, "lstm_sizes      :", lstmSizes)
print('#' * 5, "epochs          :", epochs)
print('#' * 5, "learning_rate   :", lr)
print('#' * 5, "batchSize       :", batchSize)

###############
# data pre-processing
trainEssay, trainScore, validEssay, validScore = get_data(40, 20)

vectorTrainEssay, lengthsTrainEssay = embedding_parag(trainEssay)
vectorValidEssay, lengthsValidEssay = embedding_parag(validEssay)

print("graph on")

with tf.Graph().as_default():
    # modeling
    i, l, l_p, s, bs, kp = model_inputs()
    lo, c, final = build_lstm_layers(lstmSizes, i, l, bs, kp)
    pre, loss, opt = build_cost_fn_and_opt(lo, l_p, s, lr, bs)

    loss_hist = tf.summary.scalar('loss_hist', loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        n_batches = len(vectorTrainEssay) // batchSize

        # merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('board/train', sess.graph)
        valid_writer = tf.summary.FileWriter('board/valid')

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            now_time = -time.time()
            state = sess.run(c.zero_state(batchSize, tf.float32))
            train_acc = []

            for ii, (x, ll, y) in enumerate(get_batches(vectorTrainEssay, lengthsTrainEssay, trainScore, batchSize), 1):
                llp = [[index, length - 1] for index, length in enumerate(ll)]
                feed = {i: x,
                        l: ll,
                        l_p: llp,
                        s: y,
                        bs: batchSize,
                        kp: 0.5}
                loss_, state, _ = sess.run([loss_hist, final, opt], feed_dict=feed)

                train_writer.add_summary(loss_, ii)

                ##### Validation
                if (ii + 1) % n_batches == 0:
                    val_acc = []
                    val_state = sess.run(c.zero_state(batchSize, tf.float32))
                    for xx, lll, yy in get_batches(vectorValidEssay, lengthsValidEssay, validScore, batchSize):
                        lllp = [[index, length - 1] for index, length in enumerate(lll)]
                        feed = {i: xx,
                                l: lll,
                                l_p: lllp,
                                s: yy,
                                bs: batchSize,
                                kp: 1}
                        val_loss, val_state = sess.run([loss_hist, final], feed_dict=feed)

                        # val_acc.append(val_loss)
                        valid_writer.add_summary(val_loss, ii)

                    now_time += time.time()
                    now_time = time.gmtime(now_time)
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Batch: {}/{}...".format(ii + 1, n_batches),
                          "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))
        saver.save(sess, "logic_models/dongs", global_step=globalStep)

        train_writer.close()
        valid_writer.close()
