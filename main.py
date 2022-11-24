from question_answer import answer_question
from small_talk import replicate_answer, find_response
from name import get_name_similarity
from identity_management import set_username, is_name_change
from joblib import load
from pprint import pprint
from process_text import preprocess_text

#   (5 total features to pick from: 50% rule dictates that 3 features must be picked)
#
#   Core features:
#   Intent Matching: Use intent matching to distinguish whether something is a question or small talk
#   Question Answering: Check similarity of input to the dataset for q&a and then return the answer of that question.
#   Small talk: Check similarity of input to the dataset for small talk and then return the response for that input.

NAME = "NAME"
SMALLTALK = "SMALL TALK"
QUESTION = "QUESTION"

qa_data = load("./joblibs/qa_dataset.joblib")
name_data = load("./joblibs/name_dataset.joblib")
intent_data = load("./joblibs/intent_dataset.joblib")

user_name = "User"

print("\nChatbot: Hi, I'm Steve. What is your name?\n")
user_input = input(f"{user_name}: ").lower()
user_name = set_username(user_input)
print(
    f"Steve: Let me know at any time if you want to change your username {user_name} :)")
print("Steve: You can either chat with me or ask questions about:\n -- University, Youtube, Humanism, Geological History of Earth, Police, Infection, Hunting --\n")

query = "TEMP STRING"
while (query):
    q_input = input(user_name + ": ")
    # If it is not question and answer. We give our chatbot a slightly more human response to give it more context.
    query = preprocess_text(text=q_input, type="lemmatisation")
    
    intent = get_name_similarity(name_data, query)
    if intent == NAME:
        if (is_name_change(query)):
            print(f"Steve: {user_name}, please enter a new name")
            user_input = input(user_name + ": ").lower()
            user_name = set_username(user_input)
            print(f"Steve: Congratulations your name is now {user_name}")
            continue
        else:
            print(f"Steve: Your name is {user_name}")
            continue
    
    # Extracts only the relevant key words for the question and answer
    qa_query = preprocess_text(text=q_input, stopwords = True, type="lemmatisation")
    response = answer_question(qa_data, qa_query)
    if (response != "NOT FOUND"):
        print(f"Here is the answer to your question: {response}")
        continue


    response = find_response(intent_data, query)
    if (response != "NOT FOUND"):
        print("Steve: " + response)
        continue
    
    
    
    print("I'm sorry, I don't understand what you are saying. Could you try rephrasing?")


print("Steve: I enjoyed talking to you :)")

# %%
