import pandas as pd
import numpy as np
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

def get_data(paths=[train_data_path,valid_data_path,valid_predic_path]):
    pd_train = pd.read_csv(paths[0], sep='\t')
    pd_valid_data = pd.read_csv(paths[1], sep='\t')
    pd_valid_y = pd.read_csv(paths[2])
    pd_valid = pd_valid_data.merge(pd_valid_y.rename(columns={'prediction_id':'domain1_predictionid'}))

    pd_train = pd_train[['essay_set', 'essay', 'domain1_score']].rename(columns={'domain1_score':'score'})
    pd_valid = pd_valid[['essay_set', 'essay', 'predicted_score']].rename(columns={'predicted_score':'score'})

    train_x = np.array(pd_train['essay'])
    train_y = [ get_nom_score(p['essay_set'], p['score']) for _, p in pd_train.iterrows() ]
    valid_x = np.array(pd_valid['essay'])
    valid_y = [ get_nom_score(p['essay_set'], p['score']) for _, p in pd_valid.iterrows() ]

    return train_x, train_y, valid_x, valid_y
