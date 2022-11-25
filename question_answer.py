import pandas as pd
import numpy as np
from match_intent import calculate_similarity
from joblib import load

QA_THRESHOLD = 0.7

class QuestionAnswer():

    def __init__(self) -> None:
        self.qa_data = load("./joblibs/qa_dataset.joblib")

    def answer_question(self, query):
        cos = calculate_similarity(self.qa_data, query)
        if cos.max() >= QA_THRESHOLD:
            id_argmax = np.where(cos == np.max(cos, axis=0))
            id = np.random.choice(id_argmax[0])
            return (self.qa_data['text'].loc[id])
        else:
            return ('NOT FOUND')

