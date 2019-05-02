import time, sys, os, json
import tensorflow as tf
import pymongo
from bson.objectid import ObjectId
from embedding import *

# modify current working directory
os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))

# init..?
essayId = sys.argv[1]
modelDirPath = 'logic_models'
_start_tm = time.time()
ret = dict()
ret['result'] = 0

# load essay MongoDB
conn = pymongo.MongoClient('localhost')
db = conn.get_database('mongodb_tutorial')
essayCollection = db.get_collection('essays')
try:
    result = essayCollection.find({"_id": ObjectId(essayId)})[0]
except IndexError:
    print(json.dumps(ret))
    exit(0)

ret['result'] = 1
inputEssay, length_essay = embedding_parag(result.get('paragraph', 'Hi~'))
inputOpinion = result.get('opinion', 'Hi~')
nameAuthor = result.get('author', 'customer')

checkpoint_file = tf.train.latest_checkpoint(modelDirPath)

graph_ = tf.Graph()
with graph_.as_default():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        res = graph_.get_tensor_by_name("predictions:0")
        i = graph_.get_tensor_by_name('essays:0')
        l = graph_.get_tensor_by_name('essay_lengths:0')
        lp = graph_.get_tensor_by_name('indice:0')
        kp = graph_.get_tensor_by_name('keep_prob:0')

        siba = sess.run(res, feed_dict={
            i: [inputEssay],
            l: [length_essay],
            lp: [[0, length_essay-1]],
            kp: 1,
        })

_running_tm = time.gmtime(time.time()-_start_tm)
ret['score'] = siba[0][0] * 100
ret['time'] = time.time()-_start_tm
print(json.dumps(ret))
