import tensorflow as tf
import argparse
from nltk.tokenize import sent_tokenize
import time

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--essay-path", dest="essay", type=str, metavar='<str>', required=True, help="The path to the essay")
parser.add_argument("-s","--step", dest="step", type=str, metavar='<str>', default=3, help="saver global step number (default=3)")
args = parser.parse_args()

start_time = time.time()
essay_path = args.essay
model_dir = "logic_models"
global_step = args.step
meta_path = model_dir + "/dongs-"+global_step+".meta"

with open(essay_path, 'r') as f:
    input_essay = f.read()

input_essay = sent_tokenize(input_essay)
essay_len = len(input_essay)

# padding
for _ in range(100-essay_len):
    input_essay.append("")

input_essay = [input_essay for _ in range(10)]
essay_len = [essay_len for _ in range(10)]

with tf.Graph().as_default():
    saver = tf.train.import_meta_graph(meta_path)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        graph = tf.get_default_graph()
        essay = graph.get_tensor_by_name("inputs:0")
        lens = graph.get_tensor_by_name("lengths:0")
        kp = graph.get_tensor_by_name("keep_prob:0")
        prediction = graph.get_tensor_by_name("predictions:0")
        ret = sess.run(prediction, feed_dict={essay:input_essay, kp:1, lens: essay_len})

print(ret[0][0])

run_time = time.gmtime(start_time-time.time())
print("{} min {} sec".format(run_time.tm_min, run_time.tm_sec))
