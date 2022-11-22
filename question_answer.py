import pandas as pd
import numpy as np
from match_intent import calculate_similarity

QA_THRESHOLD = 0.6

def answer_question(qa_data, query):
    cos = calculate_similarity(qa_data, query)

    if cos.max() >= QA_THRESHOLD:
        id_argmax = np.where(cos == np.max(cos, axis=0))
        id = np.random.choice(id_argmax[0])
        return (qa_data['text'].loc[id])
    else:
        return ('NOT FOUND')

