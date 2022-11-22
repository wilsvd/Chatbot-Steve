from match_intent import calculate_similarity
import numpy as np

NAME_THRESHOLD = 0.8

def get_name_similarity(name_data, query):
    cos = calculate_similarity(name_data, query)
    if cos.max() >= NAME_THRESHOLD:
        return "NAME"
    else:
        return "NOT FOUND"