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
parser.add_argument("-ds", "--data-set", dest="cntDataset", type=int, metavar='<int>', default=20,
                    help="Number of data set (default=20)")
args = parser.parse_args()

###############
# Hyper params
global_step = args.step
lstm_size = [1024, 256]
epochs = args.epochs
lr = 0.01
batch_size_ = 100
dataset_cnt = args.cntDataset

print('#' * 5, "Global Step     :", global_step)
print('#' * 5, "LSTM Cell Size  :", lstm_size)
print('#' * 5, "Epochs          :", epochs)
print('#' * 5, "Learning Rate   :", lr)
print('#' * 5, "Batch Size      :", batch_size_)
print('#' * 5, "Data Set Count  :", dataset_cnt)

print("graph on")
with tf.device("/gpu:0"):
    with tf.Graph().as_default():
        # modeling
        essays, lengths, indice, scores, batch_size, keep_prob = model_inputs()
        lstm_outputs, lstm_cell, lstm_init_state, lstm_final_state = build_lstm_layers(lstm_size, essays, lengths, batch_size, keep_prob)
        predictions, losses, optimizer = build_cost_fn_and_opt(lstm_outputs, indice, scores, lr)

        # to Tensorboard
        loss_hist = tf.summary.scalar('loss_hist', losses)
        # to saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # merged = tf.summary.merge_all()
            start_time = int(time.time())
            train_writer = tf.summary.FileWriter('board/train-'+str(start_time), sess.graph)
            # valid_writer = tf.summary.FileWriter('board/valid-'+str(start_time))

            sess.run(tf.global_variables_initializer())

            for e in range(epochs):
                now_time = -time.time()
                state = sess.run(lstm_cell.zero_state(batch_size_, tf.float32))
                for _index, (essays_, scores_) in enumerate(get_batches2(), 1):
                    lx = [len(xx) for xx in essays_]
                    llp = [[index, length - 1] for index, length in enumerate(lx)]
                    feed = {
                        essays:        essays_,
                        lengths:       lx,
                        indice:        llp,
                        scores:        [[score] for score in scores_],
                        batch_size:    batch_size_,
                        lstm_init_state: state,
                        keep_prob:     0.5
                    }
                    loss_, state, _ = sess.run([loss_hist, lstm_final_state, optimizer], feed_dict=feed)
                    if _index % 20:
                        train_writer.add_summary(loss_, _index)

                    # if _index % 10 == 0:
                    #     pdValidPath = '../data/valid_preproc_'+str(_index//10)+'.csv'
                    #     vx, vlx, vy = get_data_set(pdValidPath)
                    #     val_state = sess.run(lstmCell.zero_state(batchSize, tf.float32))
                    #     vllp = [[index, length - 1] for index, length in enumerate(vlx)]
                    #     feed = {essaysTensor:       vx,
                    #             essaysLength:       vlx,
                    #             essaysLength2:      vllp,
                    #             essaysScore:        vy,
                    #             batchSizeTensor:    batchSize,
                    #             keepProbTensor:     1}
                    #     val_loss, val_state = sess.run([loss_hist, lstmFinalState], feed_dict=feed)
                    #     valid_writer.add_summary(val_loss, _index*batchSize)

                now_time += time.time()
                now_time = time.gmtime(now_time)
                print("Epoch: {}/{}...\n".format(e + 1, epochs),
                      "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))
                saver.save(sess, "logic_models/"+str(start_time), global_step=e)

            train_writer.close()
            # valid_writer.close()
