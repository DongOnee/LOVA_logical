from embedding import *
import tensorflow as tf
import argparse
import time
from nltk.tokenize import sent_tokenize


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--essay-path", dest="essay", type=str, metavar='<str>', required=True,
                    help="The path to the essay")
parser.add_argument("-s", "--step", dest="step", type=str, metavar='<str>', default="5",
                    help="saver global step number (default=5)")
args = parser.parse_args()

start_time = time.time()
essay_path = args.essay
model_dir = "logic_models"
global_step = args.step
meta_path = model_dir + "/dongs-"+global_step+".meta"

with open(essay_path, 'r') as f:
    input_essay = f.read()

input_essay = [sent_tokenize(input_essay)]

vector_essay, length_essay = embedding_parag(input_essay)

with tf.Graph().as_default():
    saver = tf.train.import_meta_graph(meta_path)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        graph_ = tf.get_default_graph()
        res = graph_.get_tensor_by_name("result:0")
        i = graph_.get_tensor_by_name('inputs:0')
        l = graph_.get_tensor_by_name('lengths:0')
        lp = graph_.get_tensor_by_name('lengths_pad:0')
        kp = graph_.get_tensor_by_name('keep_prob:0')
        bs = graph_.get_tensor_by_name('batch_size:0')
        siba = sess.run(res, feed_dict={
            i: vector_essay,
            l: length_essay,
            lp: [[0, length_essay[0]]],
            kp: 1,
            bs: 1
        })

print(siba[0][0])
run_time = time.gmtime(time.time()-start_time)
print("{} min {} sec".format(run_time.tm_min, run_time.tm_sec))
