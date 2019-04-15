import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import time

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

paths = dataPath

pdTrain = pd.read_csv(paths[0], sep='\t')
batchCnt = len(pdTrain)//batchSize
pdTrain = pdTrain[:batchSize*batchCnt]
pdTrain = pdTrain[['essay_set', 'essay', 'domain1_score']]\
    .rename(columns={'domain1_score': 'score'})\
    .sample(frac=1)\
    .reset_index(drop=True)
trainX = np.array(pdTrain['essay'])
trainX = [sent_tokenize(pa) for pa in trainX]
trainY = [get_nom_score(p['essay_set'], p['score']) for _, p in pdTrain.iterrows()]

# pd_valid_data = pd.read_csv(paths[1], sep='\t')
# pd_valid_y = pd.read_csv(paths[2])
# pdValid = pd_valid_data.merge(pd_valid_y.rename(columns={'prediction_id': 'domain1_predictionid'}))
# batchCnt = len(pdValid)//batchSize
# pdValid = pdValid[:batchSize*batchCnt]
# pdValid = pdValid[['essay_set', 'essay', 'predicted_score']]\
#     .rename(columns={'predicted_score': 'score'})\
#     .sample(frac=1)\
#     .reset_index(drop=True)
# validX = np.array(pdValid['essay'])
# validX = [sent_tokenize(pa) for pa in validX]
# validY = [get_nom_score(p['essay_set'], p['score']) for _, p in pdValid.iterrows()]

with tf.device("/gpu:0"):
    with tf.Graph().as_default():
        elmo_module_url = "https://tfhub.dev/google/elmo/2"
        embed = hub.Module(elmo_module_url)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(batchCnt):
                print("{}'th batch".format(i+1))
                now_time = -time.time()
                xx = trainX[i*batchSize:i*batchSize+batchSize]
                yy = trainY[i*batchSize:i*batchSize+batchSize]
                df = pd.DataFrame(columns=['essay', 'score'])
                for ii, (xxx, yyy) in enumerate(zip(xx, yy)):
                    embedding = embed(xxx, signature="default", as_dict=True)['elmo']
                    sentence_rep = tf.reduce_mean(embedding, 1)  # [??, ???, 1024] => [??, 1024]
                    df.loc[ii] = [sess.run(sentence_rep).tolist(), yyy]
                df.to_csv('../preproc/train_preproc_' + str(i) + '.csv', index=False)
                now_time += time.time()
                now_time = time.gmtime(now_time)
                print("Count: {}/{}...".format(i, batchCnt),
                      "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))
                del xx, yy, df, embedding, sentence_rep
