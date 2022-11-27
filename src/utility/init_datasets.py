import yaml
import json
import numpy as np
import pandas as pd
from joblib import dump
from utility.process_text import ProcessText

QA_FILE = "../datasets/qa_data.json"
NAME_FILE = "../datasets/name_data.yml"
INTENT_FILE = "../datasets/sm_intents.csv"

class DatasetInitializer():

    def __init__(self) -> None:
        self.process_text = ProcessText()

    def setup_qa_dataset(self):
        file = json.loads(open(QA_FILE).read())
        record_path = ["data", "paragraphs", "qas", "answers"]
        # Remove the nesting of the data
        bot_nest = pd.json_normalize(file, record_path[0:len(record_path)])
        mid_nest = pd.json_normalize(file, record_path[:-1])
        # Merge dataframes
        ndx = np.repeat(mid_nest['id'].values, mid_nest['answers'].str.len())
        bot_nest['q_idx'] = ndx
        qa_dataset = pd.concat([mid_nest[['id', 'question']].set_index(
            'id'), bot_nest.set_index('q_idx')], axis=1, sort=False).reset_index()
        qa_dataset = qa_dataset.drop(columns=['answer_start', 'index'])

        qa_dataset['question'] = qa_dataset['question'].apply(self.process_text.preprocess_text, stopwords=False, type='lemmatisation')
        dump(qa_dataset, "../joblibs/qa_dataset.joblib")

    def setup_intent_sm_dataset(self):
        intent_dataset = pd.read_csv(INTENT_FILE)
        intent_dataset['Utterances'] = intent_dataset['Utterances'].apply(self.process_text.preprocess_text, stopwords=False, type='lemmatisation')
        dump(intent_dataset, "../joblibs/intent_dataset.joblib")

    def setup_name_dataset(self):
        name_dataset = pd.DataFrame(columns=['question'])
        docs = yaml.safe_load(open(NAME_FILE).read())
        questions = docs['question']
        for question in questions:
            name_dataset.loc[len(name_dataset.index)] = [question[0]]

        dump(name_dataset, "../joblibs/name_dataset.joblib")


data_initializer = DatasetInitializer()
data_initializer.setup_qa_dataset()
data_initializer.setup_name_dataset()
data_initializer.setup_intent_sm_dataset()
