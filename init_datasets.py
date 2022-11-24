import yaml
import os
import json
import numpy as np
import pandas as pd
from joblib import dump

from process_text import preprocess_text

QA_FILE = "./datasets/stanford/train-v1.1.json"
SM_FOLDER = "./datasets/small_talk/"
NAME_FILE = "./datasets/name/name.yml"
INTENT_FILE = "./datasets/small_talk/Small_talk_Intent.csv"


def setup_qa_dataset():
    file = json.loads(open(QA_FILE).read())
    record_path = ["data", "paragraphs", "qas", "answers"]
    # Remove the nesting of the data
    bot_nest = pd.json_normalize(file, record_path[0:len(record_path)])
    mid_nest = pd.json_normalize(file, record_path[:-1])
    top_nest = pd.json_normalize(file, record_path[:-2])
    # Merge dataframes
    # idx = np.repeat(top_nest['context'].values, top_nest.qas.str.len())
    ndx = np.repeat(mid_nest['id'].values, mid_nest['answers'].str.len())
    # mid_nest['context'] = idx
    bot_nest['q_idx'] = ndx
    # %%
    qa_dataset = pd.concat([mid_nest[['id', 'question']].set_index(
        'id'), bot_nest.set_index('q_idx')], axis=1, sort=False).reset_index()
    qa_dataset = qa_dataset.drop(columns=['answer_start', 'index'])

    # qa_dataset['question'] = qa_dataset['question'].apply(preprocess_text, type='lemmatisation')
    dump(qa_dataset, "./joblibs/qa_dataset.joblib")


def setup_small_talk_dataset():
    sm_dataset = pd.DataFrame(columns=['question', 'answer'])
    for dirname, _, filenames in os.walk(SM_FOLDER):
        for filename in filenames:
            file = open(os.path.join(dirname, filename), 'rb')
            docs = yaml.safe_load(file)
            conversations = docs['conversations']
            for con in conversations:
                if len(con) > 2:
                    replies = con[1:]
                    ans = ""
                    for rep in replies:
                        ans += ' '+rep
                    sm_dataset.loc[len(sm_dataset.index)] = [con[0], ans]
                elif len(con) > 1:
                    sm_dataset.loc[len(sm_dataset.index)] = [con[0], con[1]]

    # sm_dataset['question'] = sm_dataset['question'].apply(preprocess_text, type='lemmatisation')
    dump(sm_dataset, "./joblibs/sm_dataset.joblib")

def setup_intent_sm_dataset():
    intent_dataset = pd.read_csv(INTENT_FILE)
    dump(intent_dataset, "./joblibs/intent_dataset.joblib")


def setup_name_dataset():
    name_dataset = pd.DataFrame(columns=['question'])
    docs = yaml.safe_load(open(NAME_FILE).read())
    questions = docs['question']
    for question in questions:
        name_dataset.loc[len(name_dataset.index)] = [question[0]]

    # name_dataset['question'] = name_dataset['question'].apply(preprocess_text, type='lemmatisation')
    dump(name_dataset, "./joblibs/name_dataset.joblib")


setup_qa_dataset()
# setup_small_talk_dataset()
setup_name_dataset()
setup_intent_sm_dataset()
