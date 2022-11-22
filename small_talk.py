import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from process_text import create_sentence
THRESHOLD = 0.4

def make_small_talk(sm_data, similarity):
    cos = similarity

    if cos.max() >= THRESHOLD:
        id_argmax = np.where(cos == np.max(cos, axis=0))
        id = np.random.choice(id_argmax[0])
        result = sm_data['answer'].loc[id]
        return result
    else:
        return 'NOT FOUND'


def replicate_answer(query):
    new_sentence = create_sentence(query)
    return new_sentence
