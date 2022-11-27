from features.question_answer import QuestionAnswer

class TestQuestionAnswer():

    def __init__(self) -> None:
        self.q_a = QuestionAnswer()
        self.qa_dataset = self.q_a.get_qa_data()

    def test_accuracy(self):
        correct_count = 0
        total_count = 0
        for key, question in enumerate(self.qa_dataset['question']):

            response = self.q_a.get_top_5_similar(question)
            bestResult = response[key]

            res = self.qa_dataset['text'].loc[key]
            if bestResult == res:
                correct_count += 1
            total_count += 1

        print(correct_count)
        print(total_count)

TestQuestionAnswer().test_accuracy()