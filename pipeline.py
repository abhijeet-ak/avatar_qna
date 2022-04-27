# __author__ Abhijeet Shrivastava
# __ email__ abhijeet@akaiketech.com

import json
from asr import AutomaticSpeechRecogModel
from ss import SpeechSynthesisModel
from qna import QnAModel


class QnAAvatar:
    def __init__(self, context_json: str):
        self.asr_model = AutomaticSpeechRecogModel()
        self.qna_model = QnAModel()
        self.ss_model = SpeechSynthesisModel()
        self.context = json.load(open(context_json, 'rb'))

    def get_context(self):
        return self.context[0]['paragraphs'][0]['context']

    # set_context change_knoledgebase
    def set_context(self, context_data):
        self.context[0]['paragraphs'][0]['context'] = context_data

    def clear_questions(self):
        self.context[0]['paragraphs'][0]['qas'] = []

    def set_questions(self, transcriptions):
        for question, qtext in transcriptions.items():
            self.context[0]['paragraphs'][0]['qas'].append({'id': question, 'question': qtext})

    def execute_pipeline(self, file_names: list):
        transcribed_text = self.asr_model.predict(file_names)

        # setting questions
        self.set_questions(transcribed_text)

        # todo enable batched question-answering
        qa_pair, qa_nbest = self.qna_model.predict_json(self.context)

        # clearing questions
        self.clear_questions()

        audios = []  # todo create batched ss prediction
        for questions in transcribed_text.keys():
            audios.append(self.ss_model.predict(qa_pair[questions][1]))
        return audios

    def speech_to_text(self, file_name):
        transcribed_text = self.asr_model.predict(file_name)
        return transcribed_text

    def answer_question(self, transcribed_text):
        self.set_questions(transcribed_text)
        qa_pair = self.qna_model.predict_json(self.context)
        self.clear_questions()
        return qa_pair

    def speak_answer(self, qa_pair):
        audios = []
        for questions in qa_pair.keys():
            audios.append(self.ss_model.predict(qa_pair[questions][1]))
        return audios


if __name__ == '__main__':
    avatar = QnAAvatar('sample.json')
    tt = avatar.speech_to_text(['q1.wav'])
    avatar.answer_question(tt)
    print('asdf')
