import pandas as pd
import glob
from multiprocessing import Pool
import tensorflow as tf
from models import embedding_layer
from nltk.tokenize import sent_tokenize


def get_batches(train_or_valid="train", batch_size=100):
    """
    :param train_or_valid: to select train data or valid data
    :param batch_size: slice batchsize
    :return: yield method essays, lengths, scores
    """
    filepaths = glob.glob("../preproc3/"+train_or_valid+"_preproc_*")

    essays, lengths, scores = list(), list(), list()
    for count, filepath in enumerate(filepaths, 1):
        tmp = pd.read_csv(filepath).values
        x = tmp[:100]
        essays.append(x)
        length = tmp[-1, 0]
        lengths.append(length)
        y = tmp[-1, 1]
        del [[tmp]]
        scores.append(y)
        if count % batch_size == 0:
            yield essays, lengths, scores
            essays.clear()
            lengths.clear()
            scores.clear()


def parallelize_dataframe(train_or_valid="train", batch_size=100):
    """
    USE MULTIPROCESSING
    :param train_or_valid: to select train data or valid data
    :param batch_size: slice batchsize
    :return: yield method essays, lengths, scores
    """
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


def embedding_parag(input_paragraphs):
    """
    :param input_paragraphs: list of paragraphs
    :return: list of preprocessed paragraphs, list of number of paragraphs
    """
    preprocessed, sentence_len = list(), list()

    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            sentences, embeddings = embedding_layer()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for paragraph in input_paragraphs:
                    paragraph = sent_tokenize(paragraph)
                    length = len(paragraph)
                    sentence_len.append(length)
                    sentence_rep = sess.run(embeddings, feed_dict={sentences: paragraph})
                    pad = [[0] * 1024]*100
                    pad[:length] = sentence_rep.tolist()
                    preprocessed.append(pad)

    return preprocessed, sentence_len
