from features.small_talk import SmallTalk
from features.question_answer import QuestionAnswer
from features.identity_management import IdentityManagement
from utility.process_text import ProcessText

# import nltk


# DONE PROCESS
small_talk = SmallTalk()
q_a = QuestionAnswer()
id_manage = IdentityManagement()
process_text = ProcessText()

if __name__ == "__main__":
    # nltk.download("stopwords")
    # nltk.download('wordnet')
    # nltk.download('universal_tagset')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('punkt')
    # nltk.download('omw-1.4')

    print("\nChatbot: Hi, I'm Steve. What is your name?\n")
    user_input = input(f"User: ").lower()
    user_name = id_manage.set_username(user_input)
    print(
        f"Steve: Let me know at any time if you want to change your username {user_name} :)"
    )
    print(
        "Steve: You can either chat with me or ask questions about:\n -- University, Youtube, Humanism, Geological History of Earth, Police, Infection, Hunting --\n"
    )
    print("Steve: If you want to end the conversation at any time just say Bye \n")

    query = "TEMP STRING"
    while query:
        q_input = input(user_name + ": ")
        query = process_text.preprocess_text(text=q_input, type="lemmatisation")

        response = q_a.answer_question(query)
        if response != "NOT FOUND":
            print(f"Steve: Here is the answer to your question --- {response}")
            continue

        response = id_manage.get_name_similarity(query)
        if response != "NOT FOUND":
            if id_manage.is_name_change(query):
                print(f"Steve: {user_name}, please enter a new name")
                user_input = input(user_name + ": ").lower()
                user_name = id_manage.set_username(user_input)
                print(f"Steve: Congratulations your name is now {user_name}")
                continue
            else:
                print(f"Steve: Your name is {user_name}")
                continue

        response = small_talk.find_response(query)
        print(f"Steve: {response}")

        if query.lower() == "bye":
            break

    print("Steve: I enjoyed talking to you :)")
