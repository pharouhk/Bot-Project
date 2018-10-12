import nltk
import pickle
import re
import numpy as np
import gensim
from gensim.models import KeyedVectors

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_model.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags_2',
    'WORD_EMBEDDINGS': 'word_embeddings_reduced.pkl',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    wv_embeddings = unpickle_file(embeddings_path)
    
    dim = 25
    return wv_embeddings, dim

        

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function n the 3rd assignment.

    known = []
    all_vec = []
    if len(question.split()) == 0:
        return np.zeros(25)
    elif len(question.split()) == 1 and question not in embeddings:
        return np.zeros(25)
    else: 
        for i in question.split():
            if i not in embeddings:
                all_vec.append(np.zeros(25))
            elif embeddings[i] == []:
                next(question.split())
            else:
                known.append(i)
                each_vec = embeddings[i]
                all_vec.append(each_vec.reshape(1,-1))
    if len(known) > 0:
        return (sum(all_vec))/ len(known)
    else:
        return (sum(all_vec))/ len(all_vec)

    
def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
