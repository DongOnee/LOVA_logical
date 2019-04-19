import pandas as pd
import numpy as np
import time
import gc
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


def make_model(url):
    input_sent = tf.placeholder(tf.string, [None], name='input')
    elmo_module = hub.Module(url, trainable=False)
    embedding = elmo_module(input_sent, signature="default", as_dict=True)['default']
    return input_sent, embedding


def get_nom_score(prompt_id, scores_):
    min_, max_ = asap_ranges[prompt_id]
    return (scores_-min_) / (max_ - min_)


def get_train_data(path, batch_size):
    _df = pd.read_csv(path, sep='\t')
    batch_cnt = len(_df) // batch_size
    _df = _df[:batch_size * batch_cnt]
    _df = _df[['essay_set', 'essay', 'domain1_score']]\
        .rename(columns={'domain1_score': 'score'})\
        .sample(frac=1)\
        .reset_index(drop=True)
    _essays = np.array(_df['essay'])
    _essays = [sent_tokenize(_essay) for _essay in _essays]
    _scores = [get_nom_score(_row['essay_set'], _row['score']) for _, _row in _df.iterrows()]
    del [[_df]]
    gc.collect()
    for cnt in range(batch_cnt):
        yield _essays[cnt * batch_size:(cnt+1) * batch_size], _scores[cnt * batch_size:(cnt+1) * batch_size]


def get_valid_data(path_essay, path_score, batch_size):
    _df_essays = pd.read_csv(path_essay, sep='\t')
    _df_scores = pd.read_csv(path_score)
    _df = _df_essays.merge(_df_scores.rename(columns={'prediction_id': 'domain1_predictionid'}))
    batch_cnt = len(_df) // batch_size
    _df = _df[:batch_size * batch_cnt]
    _df = _df[['essay_set', 'essay', 'predicted_score']] \
        .rename(columns={'predicted_score': 'score'}) \
        .sample(frac=1) \
        .reset_index(drop=True)
    _essays = np.array(_df['essay'])
    _essays = [sent_tokenize(_essay) for _essay in _essays]
    _scores = [get_nom_score(_row['essay_set'], _row['score']) for _, _row in _df.iterrows()]
    del [[_df_essays, _df_scores, _df]]
    gc.collect()
    for cnt in range(batch_cnt):
        yield _essays[cnt * batch_size:(cnt+1) * batch_size], _scores[cnt * batch_size:(cnt+1) * batch_size]


if __name__ == '__main__':
    batch_size_ = 100

    with tf.device("/cpu:0"):
        with tf.Graph().as_default():
            elmo_module_url = "https://tfhub.dev/google/elmo/2"
            sentences, embeddings = make_model(elmo_module_url)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                print("Train Data Preprocessing")
                for batch_count, (essays, scores) in enumerate(get_train_data(paths[0], batch_size_), 1):
                    now_time = -time.time()
                    df = pd.DataFrame(columns=['essay', 'lengths', 'score'])
                    for i, (essay, score) in enumerate(zip(essays, scores), 1):
                        sentence_rep = sess.run(embeddings, feed_dict={sentences: essay})
                        df.loc[i] = [sentence_rep.tolist(), len(essay), score]
                    df.to_csv('../preproc/train_preproc_' + str(batch_count).zfill(4) + '.csv', index=False)
                    del [[df]]
                    gc.collect()
                    now_time += time.time()
                    now_time = time.gmtime(now_time)
                    print("Count: {}...\n".format(batch_count),
                          "Time: {}hour {}min {}sec...".format(now_time.tm_hour, now_time.tm_min, now_time.tm_sec))

                print("Valid Data Preprocessing")
                # for batch_count, (essays, scores) in enumerate(get_valid_data(paths[1], paths[2], batch_size_), 1):
                #     make_csv(batch_count, essays, scores, "valid")
