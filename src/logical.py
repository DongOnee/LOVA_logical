import tensorflow as tf
import time, sys, os, json
import pymongo
from bson.objectid import ObjectId
from utils import embedding_parag
from coreference import coreference


def load_db(obj_id):
    conn = pymongo.MongoClient('localhost')
    db = conn.get_database('mongodb_tutorial')
    essay_collection = db.get_collection('essays')
    result = essay_collection.find_one({"_id": ObjectId(obj_id)})

    return result


def logical_validation(obj_id, model_dir_path = 'logic_models'):
    # init
    _start_tm = time.time()
    return_dict = dict()
    return_dict['result'] = 0

    load_result = load_db(obj_id)
    if load_result is None:
        print(json.dumps(return_dict))
        return return_dict

    return_dict['result'] = 1
    input_paragraph = load_result.get('paragraph', 'Hi~')
    input_paragraph = coreference(input_paragraph)
    input_essay, length_essay = embedding_parag([input_paragraph])

    with tf.device("/gpu:0"):
        graph_ = tf.Graph()
        with graph_.as_default():
            with tf.Session() as sess:
                checkpoint_file = tf.train.latest_checkpoint(model_dir_path)
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                prediction = graph_.get_tensor_by_name("predictions:0")
                essayTensor = graph_.get_tensor_by_name('essays:0')
                lengthTensor = graph_.get_tensor_by_name('essay_lengths:0')
                indexTensor = graph_.get_tensor_by_name('indice:0')
                keepTensor = graph_.get_tensor_by_name('keep_prob:0')

                score = sess.run(prediction, feed_dict={
                    essayTensor: input_essay,
                    lengthTensor: length_essay,
                    indexTensor: [[0, length_essay[0]-1]],
                    keepTensor: 1
                })

    _running_tm = time.gmtime(time.time()-_start_tm)
    return_dict['score'] = score[0][0] * 100
    return_dict['time'] = time.time()-_start_tm
    
    return return_dict


if __name__ == "__main__":
    # modify current working director
    os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))
    print(json.dumps(logical_validation(sys.argv[1])))
