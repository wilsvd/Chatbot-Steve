# %%
from process_text import preprocess_text
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import json
import numpy as np
import pandas as pd
from pprint import pprint


# %%
import yaml
import os
sm_dataset = pd.DataFrame(columns=['question', 'answer'])
# %%

for dirname, _, filenames in os.walk('./datasets/small_talk'):
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

# %%
sm_dataset
# %%
sm_dataset['question'] = sm_dataset['question'].apply(preprocess_text, type='lemmatisation')
# %%
sm_dataset
# appended_questions
# %%
QA_FILE = "./datasets/Stanford/train-v1.1.json"

print("Setting up the Q&A dataset")
file = json.loads(open(QA_FILE).read())
record_path = ["data", "paragraphs", "qas", "answers"]
# Remove the nesting of the data
bot_nest = pd.json_normalize(file, record_path[0:len(record_path)])
mid_nest = pd.json_normalize(file, record_path[:-1])
top_nest = pd.json_normalize(file, record_path[:-2])
# Merge dataframes
idx = np.repeat(top_nest['context'].values, top_nest.qas.str.len())
ndx = np.repeat(mid_nest['id'].values, mid_nest['answers'].str.len())
mid_nest['context'] = idx
bot_nest['q_idx'] = ndx
main = pd.concat([mid_nest[['id', 'question', 'context']].set_index(
    'id'), bot_nest.set_index('q_idx')], axis=1, sort=False).reset_index()

main = main.drop(columns=['answer_start', 'index'])
print("Done setting up Q&A dataset")
# %%
main

# %%


query = input("Ask me any question")
dataset = main
# %%
threshold = 0.8

embedding = TfidfVectorizer(analyzer='word')

# Similarity matching
X_tfidf = embedding.fit_transform(dataset['question']).toarray()
df_tfidf = pd.DataFrame(X_tfidf, columns=embedding.get_feature_names_out())

# process query
input_tfidf = embedding.transform([query.lower()]).toarray()

# cosine similarity
cos = 1 - pairwise_distances(df_tfidf, input_tfidf, metric='cosine')

if cos.max() >= threshold:
    id_argmax = np.where(cos == np.max(cos, axis=0))
    id = np.random.choice(id_argmax[0])
    print(dataset['text'].loc[id])
else:
    print('NOT FOUND')
# %%
