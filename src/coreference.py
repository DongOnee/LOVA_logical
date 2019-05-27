from allennlp import pretrained
import numpy as np
import re
model = pretrained.neural_coreference_resolution_lee_2017()

def coreference(paragraph):
    result = model.predict(paragraph)
    cluster = result['clusters']
    sent_token = result['document']

    cluster_word = []
    for i in range(len(cluster)):
        buffer = []
        for j in range(len(cluster[i])):
            buffer.append(' '.join(result['document'][cluster[i][j][0]:cluster[i][j][1]+1]))
        cluster_word.append(buffer)

    for item in cluster:
        main_mention = ' '.join(sent_token[item[0][0]:item[0][1]+1])
        for i in range(len(item)-1):
            if(len(sent_token[item[i+1][0]:item[i+1][1]+1]) == 1):
                sent_token = sent_token[0:item[i+1][0]]  + [main_mention] + sent_token[item[i+1][1]+1:len(sent_token)]
            else:
                pad_size = len(sent_token[item[i+1][0]:item[i+1][1]+1])-1
                padding = [' ' for i in range(pad_size)]
                sent_token = sent_token[0:item[i+1][0]]  + [main_mention] + padding + sent_token[item[i+1][1]+1:len(sent_token)]

    processed_paragraph = ' '.join(sent_token)
    processed_paragraph = re.sub(r"  ", " ", processed_paragraph)
    return processed_paragraph

#print(coreference('We are looking for a region of central Italy bordering the Adriatic Sea. The area is mostly mountainous and includes Mt. Corno, the highest peak of the mountain range. It also includes many sheep and an Italian entrepreneur has an idea about how to make a little money of them.'))