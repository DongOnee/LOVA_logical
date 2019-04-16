import numpy as np
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import pandas as pd
from ast import literal_eval
import glob

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


def get_data(numberOfTrain=-1, numberOfValid=-1, paths=None):
    if paths is None:
        paths = dataPath
    pdTrain = pd.read_csv(paths[0], sep='\t') if numberOfTrain == -1 else pd.read_csv(paths[0], sep='\t', nrows=numberOfTrain)
    pd_valid_data = pd.read_csv(paths[1], sep='\t') if numberOfValid == -1 else pd.read_csv(paths[1], sep='\t', nrows=numberOfValid)
    pd_valid_y = pd.read_csv(paths[2]) if numberOfValid == -1 else pd.read_csv(paths[2], nrows=numberOfValid)
    pd_valid = pd_valid_data.merge(pd_valid_y.rename(columns={'prediction_id': 'domain1_predictionid'}))

    pdTrain = pdTrain[['essay_set', 'essay', 'domain1_score']].rename(columns={'domain1_score': 'score'}).sample(frac=1).reset_index(drop=True)
    pd_valid = pd_valid[['essay_set', 'essay', 'predicted_score']].rename(columns={'predicted_score': 'score'}).sample(frac=1).reset_index(drop=True)

    train_x = np.array(pdTrain['essay'])
    train_y = [get_nom_score(p['essay_set'], p['score']) for _, p in pdTrain.iterrows()]
    valid_x = np.array(pd_valid['essay'])
    valid_y = [get_nom_score(p['essay_set'], p['score']) for _, p in pd_valid.iterrows()]

    train_x = [sent_tokenize(pa) for pa in train_x]
    train_y = [[pa] for pa in train_y]
    valid_x = [sent_tokenize(pa) for pa in valid_x]
    valid_y = [[pa] for pa in valid_y]

    return train_x, train_y, valid_x, valid_y


def get_batches(x, y, embed, batch_size=100):
    """
    Batch Generator for Training
    :param x            : Input array of x data
    :param y            : Input array of y data
    :param batch_size   : Input int, size of batch
    :return             : generator that returns a tuple of our x batch and y batch
    """

    n_batches = len(x)//batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        batch_x, batch_y = x[ii:ii+batch_size], y[ii:ii+batch_size]
        ret_x, ret_len = [], []
        proc_x=[]
        for paragraph in batch_x:
            embeds = embed(paragraph, signature="default", as_dict=True)['elmo']
            proc_x.append(embeds)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(batch_size):
                sentence_rep = tf.reduce_mean(proc_x[i], 1)  # [??, ???, 1024] => [??, 1024]
                length = len(batch_x[i])
                paddings = tf.constant([[0, 100 - length], [0, 0]])  # to 100 sentence padding
                sentence_rep = tf.pad(sentence_rep, paddings, "CONSTANT")  # [??, 1024] => [100, 1024]
                ret_x.append(sess.run(sentence_rep))
                ret_len.append(length)

        yield ret_x, ret_len, batch_y


def get_data_set(file_path):
    outx = [[0] * 1024 for _ in range(100)]
    x = pd.read_csv(file_path)[:]['essay'].values
    outx[:len(x)] = x
    x = [literal_eval(xx) for xx in x]
    lx = [len(xx) for xx in x]
    y = pd.read_csv(file_path)[:]['score'].values
    y = [[yy] for yy in y]

    return outx, lx, y


def get_batches2():
    filepaths = glob.glob("../data/train_preproc_*")

    for filepath in filepaths:
        tmp = pd.read_csv(filepath).values
        x = []
        for xx in tmp[:, 0]:
            pad = [[0] * 1024 for _ in range(100)]
            xx = literal_eval(xx)
            pad[:len(xx)] = xx
            x.append(pad)
        y = tmp[:, 1]
        yield x, y


def get_batches3():
    filepaths = glob.glob("../preproc/train_preproc_*")

    for filepath in filepaths:
        tmp = pd.read_csv(filepath).values
        x = []
        for xx in tmp[:, 0]:
            pad = [[0] * 1024 for _ in range(100)]
            xx = literal_eval(xx)
            pad[:len(xx)] = xx
            x.append(pad)
        lens = tmp[:, 1]
        y = tmp[:, 2]
        yield x, lens, y

