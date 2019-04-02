import argparse
from models import *
from utils import *
import tensorflow as tf
import tensorflow_hub as hub
from ast import literal_eval
import time

###############
# argument
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--step", dest="step", type=int, metavar='<int>', default=5,
                    help="saver global step number (default=5)")
parser.add_argument("-e", "--epochs", dest="epochs", type=int, metavar='<int>', default=5,
                    help="Number of epochs (default=5)")
parser.add_argument("-ds", "--data-set", dest="cntDataset", type=int, metavar='<int>', default=20,
                    help="Number of data set (default=20)")
args = parser.parse_args()

###############
# Hyper params
globalStep = args.step
epochs = args.epochs
cntDataset = args.cntDataset
lstmSizes = [1024, 256]
batchSize = 100
lr = 0.001

print('#' * 5, "global_step     :", globalStep)
print('#' * 5, "lstm_sizes      :", lstmSizes)
print('#' * 5, "epochs          :", epochs)
print('#' * 5, "learning_rate   :", lr)
print('#' * 5, "batchSize       :", batchSize)
print('#' * 5, "cntDataset      :", cntDataset)

print("graph on")
with tf.Graph().as_default():
    # modeling
    i, l, l_p, s, bs, kp = model_inputs()
    lo, c, final = build_lstm_layers(lstmSizes, i, l, bs, kp)
    pre, loss, opt = build_cost_fn_and_opt(lo, l_p, s, lr, bs)

    elmo_module_url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(elmo_module_url)

    loss_hist = tf.summary.scalar('loss_hist', loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('board/train', sess.graph)
        valid_writer = tf.summary.FileWriter('board/valid')

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            now_time = -time.time()
            state = sess.run(c.zero_state(batchSize, tf.float32))

            for cntI in range(1, cntDataset+1):
                pdDataPath = '../data/train_preproc_'+str(cntDataset*batchSize)+'.csv'
                x, lx, y = get_data_set(pdDataPath)
                llp = [[index, length - 1] for index, length in enumerate(lx)]
                feed = {i: x,
                        l: lx,
                        l_p: llp,
                        s: y,
                        bs: batchSize,
                        kp: 0.5}
                loss_, state, _ = sess.run([loss_hist, final, opt], feed_dict=feed)
                train_writer.add_summary(loss_, cntI*batchSize+ii)

                if cntI % 10 == 0:
                    pdValidPath = '../data/valid_preproc_'+str(cntI//10)+'.csv'
                    vx, vlx, vy = get_data_set(pdValidPath)
                    val_state = sess.run(c.zero_state(batchSize, tf.float32))
                    vllp = [[index, length - 1] for index, length in enumerate(vlx)]
                    feed = {i: vx,
                            l: vlx,
                            l_p: vllp,
                            s: vy,
                            bs: batchSize,
                            kp: 1}
                    val_loss, val_state = sess.run([loss_hist, final], feed_dict=feed)
                    valid_writer.add_summary(val_loss, cntI*batchSize, )

                    del vx
                    del vy

                del x
                del y

            now_time += time.time()
            now_time = time.gmtime(now_time)
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))
            saver.save(sess, "logic_models/dongs", global_step=globalStep)

            train_writer.close()
            valid_writer.close()
