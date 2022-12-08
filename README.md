# Chatbot

# University Machine

## How to Run Program

1. `cd` into `src/` directory
2. Type `python -m main`

If an error occurs it may be due to the version of Joblib. See the following instructions to update the Joblibs.

## Initialise Joblib Datasets

1. `cd` into `src/` directory
2. Type `python -m utility.init_datasets`

## Initialise Test Joblib Classifier and Vectoriser

1. `cd` into `src/` directory
2. Type `python -m tests.test_classifier`

## Referenced Code

### 1. Intent Classifer

- Inspired by [kumbaraci](https://www.kaggle.com/code/kumbaraci/intent-classification) on Kaggle.

### 2. Vectorisation

- [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

### 3. Cosine Similarity

- [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)

### 4. Text Pre-processing

- Stemmer - [NLTK](https://www.nltk.org/api/nltk.stem.snowball.html)
- Lemmatizer - [NLTK](https://www.nltk.org/_modules/nltk/stem/wordnet.html)
- Tokenizer - [NLTK](https://www.nltk.org/api/nltk.tokenize.html)
