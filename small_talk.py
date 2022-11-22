import pandas as pd
import numpy as np

from match_intent import calculate_similarity
from process_text import create_sentence
SM_THRESHOLD = 0.5

def make_small_talk(sm_data, query):
    cos = calculate_similarity(sm_data, query)

    if cos.max() >= SM_THRESHOLD:
        id_argmax = np.where(cos == np.max(cos, axis=0))
        id = np.random.choice(id_argmax[0])
        result = sm_data['answer'].loc[id]
        return result
    else:
        return 'NOT FOUND'

def replicate_answer(query):
    new_sentence = create_sentence(query)
    return new_sentence

