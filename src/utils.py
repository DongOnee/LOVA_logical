import pandas as pd
import numpy as np
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

dataPath = ['../data/training_set_rel3.tsv',
            '../data/valid_set.tsv',
            '../data/valid_sample_submission_2_column.csv']

# train_data_path = '../data/training_set_rel3.tsv'
# valid_data_path = '../data/valid_set.tsv'
# valid_predic_path = '../data/valid_sample_submission_2_column.csv'


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


def get_batches(x, lengths, y, batch_size=100):
    """
    Batch Generator for Training
    :param x            : Input array of x data
    :param lengths      : Input array of length of x data
    :param y            : Input array of y data
    :param batch_size   : Input int, size of batch
    :return             : generator that returns a tuple of our x batch and y batch
    """
    n_batches = len(x)//batch_size
    x, lengths, y = x[:n_batches*batch_size], lengths[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], lengths[ii:ii+batch_size], y[ii:ii+batch_size]
