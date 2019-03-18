import tensorflow_hub as hub
import tensorflow as tf

########################
# Model 1
#


def embedding_parag(input_paragraphs):
    """
    :param input_paragraphs: list of paragraphs
    :return: list of preprocessed paragraphs, list of number of paragraphs
    """

    elmo_module_url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(elmo_module_url)

    word_embeddings = []
    for paragraph in input_paragraphs:
        embeds = embed(paragraph, signature="default", as_dict=True)['elmo']
        word_embeddings.append(embeds)

    preprocessed = []
    sentence_len = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for paragraph in word_embeddings:
            sentence_rep = tf.reduce_mean(paragraph, 1)  # [??, ???, 1024] => [??, 1024]

            length = sess.run(tf.shape(sentence_rep)[0])  # number of sentences
            paddings = tf.constant([[0, 100 - length], [0, 0]])  # to 100 sentence padding
            sentence_rep = tf.pad(sentence_rep, paddings, "CONSTANT")  # [??, 1024] => [100, 1024]

            preprocessed.append(sess.run(sentence_rep))
            sentence_len.append(length)

    return preprocessed, sentence_len
