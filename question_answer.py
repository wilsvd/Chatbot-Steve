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
        print(cos.max())
        if cos.max() >= QA_THRESHOLD:
            id_argmax = np.where(cos == np.max(cos, axis=0))
            id = np.random.choice(id_argmax[0])
            print(self.qa_data['question'].loc[id])
            return (self.qa_data['text'].loc[id])
        else:
            return ('NOT FOUND')

