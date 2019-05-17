import tensorflow as tf
import time, sys, os, json
import pymongo
from bson.objectid import ObjectId
from utils import embedding_parag

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
result = essayCollection.find_one({"_id": ObjectId(essayId)})

if result.count() == 0:
    print(json.dumps(ret))
    exit(0)

ret['result'] = 1
inputParagraph = result.get('paragraph', 'Hi~')
# inputOpinion = result.get('opinion', 'Hi~')
# nameAuthor = result.get('author', 'customer')
inputEssay, length_essay = embedding_parag(inputParagraph)

graph_ = tf.Graph()
with tf.device("/gpu:0"):
    with graph_.as_default():
        with tf.Session() as sess:
            checkpoint_file = tf.train.latest_checkpoint(modelDirPath)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            prediction = graph_.get_tensor_by_name("predictions:0")
            essayTensor = graph_.get_tensor_by_name('essays:0')
            lengthTensor = graph_.get_tensor_by_name('essay_lengths:0')
            indexTensor = graph_.get_tensor_by_name('indice:0')
            keepTensor = graph_.get_tensor_by_name('keep_prob:0')

            score = sess.run(prediction, feed_dict={
                essayTensor: [inputEssay],
                lengthTensor: [length_essay],
                indexTensor: [[0, length_essay-1]],
                keepTensor: 1
            })

_running_tm = time.gmtime(time.time()-_start_tm)
ret['score'] = score[0][0] * 100
ret['time'] = time.time()-_start_tm
print(json.dumps(ret))
