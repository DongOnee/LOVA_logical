import pandas as pd
import numpy as np
import time
import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize

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

paths = ['../data/training_set_rel3.tsv',
         '../data/valid_set.tsv',
         '../data/valid_sample_submission_2_column.csv']


def get_nom_score(prompt_id, score):
    min_, max_ = asap_ranges[prompt_id]
    return (score-min_) / (max_ - min_)


def get_train_data(path, batch_size):
    df = pd.read_csv(path, sep='\t')
    batch_cnt = len(df) // batch_size
    df = df[:batch_size * batch_cnt]
    df = df[['essay_set', 'essay', 'domain1_score']]\
        .rename(columns={'domain1_score': 'score'})\
        .sample(frac=1)\
        .reset_index(drop=True)
    essays = np.array(df['essay'])
    essays = [sent_tokenize(_essay) for _essay in essays]
    scores = [get_nom_score(_row['essay_set'], _row['score']) for _, _row in df.iterrows()]
    for cnt in range(batch_cnt):
        yield essays[cnt * batch_size:(cnt+1) * batch_size], scores[cnt * batch_size:(cnt+1) * batch_size]


def get_valid_data(path_essay, path_score, batch_size):
    df_essays = pd.read_csv(path_essay, sep='\t')
    df_scores = pd.read_csv(path_score)
    df = df_essays.merge(df_scores.rename(columns={'prediction_id': 'domain1_predictionid'}))
    batch_cnt = len(df) // batch_size
    df = df[:batch_size * batch_cnt]
    df = df[['essay_set', 'essay', 'predicted_score']] \
        .rename(columns={'predicted_score': 'score'}) \
        .sample(frac=1) \
        .reset_index(drop=True)
    essays = np.array(df['essay'])
    essays = [sent_tokenize(_essay) for _essay in essays]
    scores = [get_nom_score(_row['essay_set'], _row['score']) for _, _row in df.iterrows()]
    for cnt in range(batch_cnt):
        yield essays[cnt * batch_size:(cnt+1) * batch_size], scores[cnt * batch_size:(cnt+1) * batch_size]


if __name__ == '__main__':
    batch_size_ = 100

    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            elmo_module_url = "https://tfhub.dev/google/elmo/2"
            embed = hub.Module(elmo_module_url)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                print("Train Data Preprocessing")
                for batch_count, (essays_, scores_) in enumerate(get_train_data(paths[0], batch_size_), 1):
                    print("{}'th batch".format(batch_count))
                    now_time = -time.time()
                    df_ = pd.DataFrame(columns=['essay', 'lengths', 'score'])
                    for index, (essay_, score_) in enumerate(zip(essays_, scores_), 1):
                        embedding = embed(essay_, signature="default", as_dict=True)['elmo']
                        sentence_rep = tf.reduce_mean(embedding, 1)  # [??, ???, 1024] => [??, 1024]
                        df_.loc[index] = [sess.run(sentence_rep).tolist(), len(essay_), score_]
                    df_.to_csv('../preproc/train_preproc_' + str(batch_count).zfill(4) + '.csv', index=False)
                    del [[df_]]
                    now_time += time.time()
                    now_time = time.gmtime(now_time)
                    print("Count: {}...\n".format(batch_count),
                          "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))

                print("Valid Data Preprocessing")
                for batch_count, (essays_, scores_) in enumerate(get_valid_data(paths[1], paths[2], batch_size_), 1):
                    print("{}'th batch".format(batch_count))
                    now_time = -time.time()
                    df_ = pd.DataFrame(columns=['essay', 'lengths', 'score'])
                    for index, (essay_, score_) in enumerate(zip(essays_, scores_), 1):
                        embedding = embed(essay_, signature="default", as_dict=True)['elmo']
                        sentence_rep = tf.reduce_mean(embedding, 1)  # [??, ???, 1024] => [??, 1024]
                        df_.loc[index] = [sess.run(sentence_rep).tolist(), len(essay_), score_]
                    df_.to_csv('../preproc/valid_preproc_' + str(batch_count).zfill(4) + '.csv', index=False)
                    del [[df_]]
                    now_time += time.time()
                    now_time = time.gmtime(now_time)
                    print("Count: {}...\n".format(batch_count),
                          "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))
