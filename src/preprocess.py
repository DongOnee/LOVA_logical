import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import sys

asap_ranges = {
    0: (0, 60),
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}

dataPath = ['../data/training_set_rel3.tsv',
            '../data/valid_set.tsv',
            '../data/valid_sample_submission_2_column.csv']


def get_nom_score(prompt_id, score):
    min_, max_ = asap_ranges[prompt_id]
    return (score-min_) / (max_ - min_)


batchSize = 100
sliceCnt = int(sys.argv[1])

paths = dataPath

pdTrain = pd.read_csv(paths[0], sep='\t', nrows=sliceCnt*batchSize)[-batchSize:]
pdTrain = pdTrain[['essay_set', 'essay', 'domain1_score']].rename(columns={'domain1_score': 'score'}).sample(frac=1).reset_index(drop=True)
trainX = np.array(pdTrain['essay'])
trainX = [sent_tokenize(pa) for pa in trainX]
trainY = [get_nom_score(p['essay_set'], p['score']) for _, p in pdTrain.iterrows()]
trainY = [[pa] for pa in trainY]

# pd_valid_data = pd.read_csv(paths[1], sep='\t', nrows=sliceCnt*batchSize)[-batchSize:]
# pd_valid_y = pd.read_csv(paths[2], nrows=sliceCnt*batchSize)[-batchSize:]
# pdValid = pd_valid_data.merge(pd_valid_y.rename(columns={'prediction_id': 'domain1_predictionid'}))
# pdValid = pdValid[['essay_set', 'essay', 'predicted_score']].rename(columns={'predicted_score': 'score'}).sample(frac=1).reset_index(drop=True)
# validX = np.array(pdValid['essay'])
# validX = [sent_tokenize(pa) for pa in validX]
# validY = [get_nom_score(p['essay_set'], p['score']) for _, p in pdValid.iterrows()]
# validY = [[pa] for pa in validY]

with tf.device("/gpu:0"):
    with tf.Graph().as_default():
        elmo_module_url = "https://tfhub.dev/google/elmo/2"
        embed = hub.Module(elmo_module_url)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ret = []
            for ii, (xx, yy) in enumerate(zip(trainX, trainY), 1):
                row = []
                embedding = embed(xx, signature="default", as_dict=True)['elmo']
                sentence_rep = tf.reduce_mean(embedding, 1)  # [??, ???, 1024] => [??, 1024]
                row.append(sess.run(sentence_rep).tolist())
                row.append(yy[0])
                ret.append(row)
                if ii % batchSize == 0:
                    print(batchSize*sliceCnt)
                    df = pd.DataFrame(ret, columns=['essay', 'score'])
                    df.to_csv('../data/train_preproc_'+str(batchSize*sliceCnt)+'.csv', index=False)
                    ret.clear()
#
#
# import tensorflow as tf
# import tensorflow_hub as hub
# import pandas as pd
# from nltk.tokenize import sent_tokenize
#
# a = pd.read_csv('../data/training_set_rel3.tsv', sep='\t', nrows=10)[:]['essay'].values
# ae = [sent_tokenize(ii) for ii in a]
#
# elmo_module_url = "https://tfhub.dev/google/elmo/2"
#
# with tf.Graph().as_default():
#     embed = hub.Module(elmo_module_url)
#     df = pd.DataFrame(columns=['essay'])
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for ii, xx in enumerate(ae):
#             embedding = embed(xx, signature="default", as_dict=True)['elmo']
#             retEmbed = sess.run(embedding)
#             df.loc[ii] = [retEmbed.tolist()]
#         df.to_csv('../data/test.csv')
#
#
# from ast import literal_eval
# b = pd.read_csv('../data/test.csv')[:]['essay'].values
# be = [literal_eval(ii) for ii in b]
