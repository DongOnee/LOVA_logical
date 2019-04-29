import numpy as np
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import pandas as pd
from ast import literal_eval
import glob
from multiprocessing import Pool
import os

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


def get_batches5(train_or_valid="train", batch_size=100):
    filepaths = glob.glob("../preproc3/"+train_or_valid+"_preproc_*")

    essays, lengths, scores = list(), list(), list()
    for count, filepath in enumerate(filepaths, 1):
        tmp = pd.read_csv(filepath).values
        x = tmp[:100]
        essays.append(x)
        len = tmp[-1, 0]
        lengths.append(len)
        y = tmp[-1, 1]
        del [[tmp]]
        scores.append(y)
        if count % batch_size == 0:
            yield essays, lengths, scores
            essays.clear()
            lengths.clear()
            scores.clear()


def parallelize_dataframe(train_or_valid="train", batch_size=100):
    num_cores = 10

    filepaths = glob.glob("../preproc3/" + train_or_valid + "_preproc_*")
    file_count = len(filepaths)
    n_batchs = file_count // batch_size
    loop_count = batch_size // num_cores
    for index_batch in range(n_batchs):
        essays_, lengths_, scores_= list(), list(), list()
        for index_loop in range(loop_count):
            # ret = [pool.apply_async(os.getpid, ()) for i in range(10)]
            ret = list()
            pool = Pool(num_cores)
            ret.extend(pool.map(load_data, filepaths[batch_size * index_batch + index_loop * num_cores:batch_size * index_batch + (index_loop+1) * num_cores]))
            pool.close()
            pool.join()
            for sibal in ret:
                essays_.append(sibal[0])
                lengths_.append(sibal[1])
                scores_.append(sibal[2])
        yield essays_, lengths_, scores_


def load_data(preproc_path):
    df = pd.read_csv(preproc_path).values
    return df[:100], df[-1, 0], df[-1, 1]
