import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
# import chatterbot
from chatterbot import ChatBot
from week5_utils import *
dialogue_list = [
]

paths = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_model.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags_2',
    'WORD_EMBEDDINGS': 'word_embeddings_reduced.pkl',
}

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        if tag_name[0] == 'c\c++':
            embeddings_path = os.path.join(self.thread_embeddings_folder, 'c_c++' + ".pkl")
        else:
            # print(tag_name)
            embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name[0] + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_thread = pairwise_distances_argmin(question_vec.reshape(1,-1), thread_embeddings, metric = 'cosine')
        
        return thread_ids[best_thread[0]]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        self.create_chitchat_bot()
    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param

        
        self.chitchat_bot = ChatBot('coursera_proj2_bot', 
            trainer='chatterbot.trainers.UbuntuCorpusTrainer', read_only = True)
        # self.chitchat_bot.train("chatterbot.corpus.english")

        # return self.chitchat_bot.get_response(cht)        
       
        

       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        from week5_utils import text_prepare
        prepared_question = [text_prepare(question)]
        features = self.tfidf_vectorizer.transform(prepared_question)
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:   
        if intent == 'dialogue':
            response = self.chitchat_bot.get_response(question)
            # chat = ChatBot('coursera_proj2_bot', trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
            # chat.train("chatterbot.corpus.english.botprofile")
            # response = chat.get_response(question) 
            return str(response)
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question[0], tag)
           
            return self.ANSWER_TEMPLATE % (tag[0], thread_id)


# if __name__ == '__main__':
#     DialogueManager(paths)