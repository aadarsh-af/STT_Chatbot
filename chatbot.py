# """
# pip install -r requirements.txt
# python chatbot.py

# """

import random
from speech_to_text import SpeechToText
from text_cleaner import Cleaner
from train_chatbot import ChatbotTrainer
from train_chatbot import ChatbotEvaluator
import time

## Driver Code
converter = SpeechToText()
cleaner = Cleaner()
trainer = ChatbotTrainer()
evaluator = ChatbotEvaluator()

bye_texts = ["Thanks! It was great talking with you!", "did you mean - \"Be with You Everytime <3\" ?", "Miss me!", "Bye! Have a great day!"]

data = trainer.load_json_data("chatbot_training_data.json")

words, classes, doc_X, doc_y = cleaner.tokenizer(data)
words = cleaner.lemmatize(words)

## To train the model
# train_X, train_y = prepare_training_data(doc_X, doc_y, words)
# model = train(train_X, train_y)

## To save the trained model
# trainer.save_my_model(model, "trained_chatbot.h5")

## To load the saved model
model = trainer.load_my_model("trained_chatbot.h5")

while True:
  text = converter.audio_to_text(dur=3)

  if text.lower() == "bye":
    print(random.choice(bye_texts))
    time.sleep(1)
    break

  predictions = evaluator.predict_class(text, words, classes, model)
  result = evaluator.get_response(predictions, data)
  print(result)