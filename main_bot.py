#!/usr/bin/env python3

import requests
import time
import logging
import argparse
import sys
import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv('.env')
# my_talk_bot = Flask(__name__)

from requests.compat import urljoin

# my_talk_bot.logger.addHandler(logging.StreamHandler(sys.stdout))
# my_talk_bot.logger.setLevel(logging.ERROR)

class BotHandler(object):
    """
        BotHandler is a class which implements all back-end of the bot.
        It has tree main functions:
            'get_updates' — checks for new messages
            'send_message' – posts new message to user
            'get_answer' — computes the most relevant on a user's question
    """

    def __init__(self, dialogue_manager):
        self.token = os.getenv('Telegram_key')
        self.api_url = "https://api.telegram.org/bot{}/".format(self.token)
        self.dialogue_manager = dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        raw_resp = requests.get(urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print("Failed to parse response {}: {}.".format(raw_resp.content, e))
            return []
        if "result" not in resp:
            return []
        
        return resp["result"]

    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        return requests.post(urljoin(self.api_url, "sendMessage"), params)



    def get_answer(self, question):
        if question == '/start':
            return "Hi, I am your project bot. How can I help you today?"
        elif question.lower() == 'hey' or question.lower() == 'hey?':
            return 'Hi, there'
        return self.dialogue_manager.generate_answer(question)


def is_unicode(text):
    return len(text) == len(text.encode())


# @my_talk_bot.route('/', methods=['GET','POST'])
def main():
    from dialogue_manager import DialogueManager

    paths = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_model.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags_2',
    'WORD_EMBEDDINGS': 'word_embeddings_reduced.pkl',
}
    advanced_manager = DialogueManager(paths)
    bot = BotHandler(advanced_manager)
    
    ###############################################################
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

    print("Ready to talk!")
    offset = 0
    while True:
        updates = bot.get_updates(offset=offset)
        for update in updates:
            print("An update received.")
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                if "text" in update["message"]:
                    text = update["message"]["text"]
                    if is_unicode(text):
                        print("Update content: {}".format(update))
                        bot.send_message(chat_id, bot.get_answer(update["message"]["text"]))
                    else:
                        bot.send_message(chat_id, "Hmm, you are sending some weird characters to me...")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)
        # logger = logging.getLogger()
        # logger.setLevel(logging.DEBUG)
if __name__ == "__main__":
    main()
    # my_talk_bot.run(host = '0.0.0.0',port=5005)
else:
    gunicorn_app = main()