import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
import csv

asap_ranges = {
    0: (0, 60),
    1: (2,12),
    2: (1,6),
    3: (0,3),
    4: (0,3),
    5: (0,4),
    6: (0,4),
    7: (0,30),
    8: (0,60)
}

train_data_path = '../data/training_set_rel3.tsv'
valid_data_path = '../data/valid_set.tsv'
valid_predic_path = '../data/valid_sample_submission_2_column.csv'

def get_nom_score(prompt_id, score):
    min_, max_ = asap_ranges[prompt_id]

    return (score-min_) / (max_ - min_)


def get_data(train_len=-1, valid_len=-1, paths=[train_data_path,valid_data_path,valid_predic_path]):
    pd_train = pd.read_csv(paths[0], sep='\t')
    pd_valid_data = pd.read_csv(paths[1], sep='\t')
    pd_valid_y = pd.read_csv(paths[2])
    pd_valid = pd_valid_data.merge(pd_valid_y.rename(columns={'prediction_id':'domain1_predictionid'}))

    if (pd_train.shape[0] < train_len):
        train_len = -1;
    if (pd_valid.shape[0] < valid_len):
        valid_len = -1;

    pd_train = pd_train[:train_len][['essay_set', 'essay', 'domain1_score']].rename(columns={'domain1_score':'score'}).sample(frac=1).reset_index(drop=True)
    pd_valid = pd_valid[:valid_len][['essay_set', 'essay', 'predicted_score']].rename(columns={'predicted_score':'score'}).sample(frac=1).reset_index(drop=True)

    train_x = np.array(pd_train['essay'])
    train_y = [ get_nom_score(p['essay_set'], p['score']) for _, p in pd_train.iterrows() ]
    valid_x = np.array(pd_valid['essay'])
    valid_y = [ get_nom_score(p['essay_set'], p['score']) for _, p in pd_valid.iterrows() ]

    train_x = [ sent_tokenize(pa) for pa in train_x]
    train_y = [ [pa] for pa in train_y]
    valid_x = [ sent_tokenize(pa) for pa in valid_x]
    valid_y = [ [pa] for pa in valid_y]

    return train_x, train_y, valid_x, valid_y


def get_batches(x, y, batch_size=100):
    """
    Batch Generator for Training
    :param x: Input array of x data
    :param y: Input array of y data
    :param batch_size: Input int, size of batch
    :return: generator that returns a tuple of our x batch and y batch
    """
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        x_batch, x_batch_len = [], []
        for sent in x[ii:ii+batch_size]: # to make 100 sentences
            x_batch_len.append(len(sent))
            sent.extend(["" for _ in range(100-len(sent))])
            x_batch.append(sent)
        yield x_batch, y[ii:ii+batch_size], x_batch_len
