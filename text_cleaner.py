import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download('omw-1.4')
import string


class Cleaner:
    """
    A class that cleans or tokenizes a given text
    """

    lemmatizer = WordNetLemmatizer()

    @staticmethod
    def tokenizer(data):
        """Tokenize a JSON data into words and classes.

        Keyword arguments:
            data -- (dict) 

        Returns:
            words, classes, doc_X, doc_y.
        """
        
        words = []
        classes = []
        doc_X = []
        doc_y = []
        
        # tokenize each pattern and append tokens to words, the patterns and the associated tag to their associated list
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                words.extend(tokens)
                doc_X.append(pattern)
                doc_y.append(intent["tag"])
            
            # add the tag to the classes if it's not there already 
            if intent["tag"] not in classes:
                classes.append(intent["tag"])
                
        classes = sorted(set(classes))

        return (words, classes, doc_X, doc_y)


    @staticmethod
    def lemmatize(words):
        """Lemmatize a list of words.

        Keyword arguments:
            words -- (list)

        Returns:
            lemmatized_words.
        """

        lemmatized_words = sorted(set([Cleaner.lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]))

        return lemmatized_words


    @staticmethod
    def clean_text(text):
        """Clean and Tokenize any text.

        Keyword arguments:
            text -- (str)

        Returns:
            lemmatized tokens of text.
        """

        tokens = nltk.word_tokenize(text)
        tokens = [Cleaner.lemmatizer.lemmatize(word) for word in tokens]
        return tokens