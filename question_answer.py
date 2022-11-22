from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np

THRESHOLD = 0.6


def answer_question(dataset, similarity):
    cos = similarity

    if cos.max() >= THRESHOLD:
        id_argmax = np.where(cos == np.max(cos, axis=0))
        id = np.random.choice(id_argmax[0])
        return (dataset['text'].loc[id])
    else:
        return ('NOT FOUND')
