# """
# Use a json file that has the structure like below...
# {
# intents:[
#     {
#     tag: "",
#     patterns: [
#         ""
#     ],
#     responses: [
#         ""
#     ]
#     },
#     ...
# ]
# }
# """

import json
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from text_cleaner import Cleaner

cleaner = Cleaner()


class ChatbotTrainer:
    """
    A class that trains a model on the provided data
    """

    @staticmethod
    def load_json_data(file_path: str) -> dict:
        """Load a JSON file as a dict.

        You can provide the JSON file path upon which a chatbot would be trained.

        Keyword arguments:
            file_path -- (str) absolute file path

        Returns:
            A dictionary that contains the JSON structured data.
        """

        with open(file_path) as datafile:
            data = json.load(datafile)

        return data


    @staticmethod
    def prepare_training_data(doc_X, doc_y, words):
        """Prepare the training data.

        Prepare training data by dividing it into train_X, train_y.

        Keyword arguments:
            doc_X -- (list) list of all patterns
            doc_y -- (list) list of all classes lined with patterns (classes may repeat)
            word -- (list) list of lemmatized words

        Returns:
            train_X and train_y.
        """

        training = []
        out_empty = [0] * len(classes)

        # creating the bag of words model
        for idx, doc in enumerate(doc_X):
            bow = []
            text = cleaner.lemmatizer.lemmatize(doc.lower())
            for word in words:
                bow.append(1) if word in text else bow.append(0)

            # mark the index of class that the current pattern is associated to
            output_row = list(out_empty)
            output_row[classes.index(doc_y[idx])] = 1

            # add the one hot encoded BoW and associated classes to training
            training.append([bow, output_row])

        # shuffle the data and convert it to an array
        random.shuffle(training)
        training = np.array(training, dtype=object)

        # split the features and target labels
        train_X = np.array(list(training[:, 0]))
        train_y = np.array(list(training[:, 1]))

        return (train_X, train_y)


    @staticmethod
    def bag_of_words(text, vocab):
        """Load a JSON file as a dict.

        You can provide the JSON file path upon which a chatbot would be trained.

        Keyword arguments:
            file_path -- (str) absolute file path

        Returns:
            A dictionary that contains the JSON structured data.
        """

        tokens = cleaner.clean_text(text)
        bow = [0] * len(vocab)
        for w in tokens:
            for idx, word in enumerate(vocab):
                if word == w:
                    bow[idx] = 1
        return np.array(bow)


    @staticmethod
    def train(train_X, train_y, epochs=90):
        """Train a Neural network classifier.

        Provide the train_X and train_y data.

        Keyword arguments:
            train_X -- (np.array) array of patterns training set
            train_y -- (np.array) array of classes training set to classify the unseen patterns

        Returns:
            A model that is also saved and can be loaded directly.
        """

        # defining some parameters
        input_shape = (len(train_X[0]),)
        output_shape = len(train_y[0])

        # the deep learning model
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation="softmax"))
        adam = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["accuracy"])
        print(model.summary())
        model.fit(x=train_X, y=train_y, epochs=epochs, verbose=1)

        return model

    
    @staticmethod
    def save_my_model(model, filename):
        """Save a keras model to file.

        Provide the trained model and give it a filename with the extension of '.h5'.

        Keyword arguments:
            model -- (keras model)
            filename -- (str)

        Returns:
            None, saves a '.h5' file to local space.
        """

        model.save(filename)
    
    
    @staticmethod
    def load_my_model(filename):
        """load a keras model to file.

        Load a model file with the extension of '.h5'.

        Keyword arguments:
            filename -- (str)

        Returns:
            model.
        """

        model = keras.models.load_model(filename)
        return model



class ChatbotEvaluator:

    @staticmethod
    def predict_class(text, vocab, labels, model):
        """Predict the class of a text.

        Provide a text, its vocab(tokendized_words), and classes.

        Keyword arguments:
            text -- (str)
            vocab -- (list)
            labels -- (list)

        Returns:
            A list of possible classes(sorted with descending order of probability) for the text.
        """

        bow = ChatbotTrainer.bag_of_words(text, vocab)
        result = model.predict(np.array([bow]))[0]
        thresh = 0.2

        y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

        y_pred.sort(key=lambda x: x[1], reverse=True)

        return_list = []

        for r in y_pred:
            return_list.append(labels[r[0]])

        return return_list


    @staticmethod
    def get_response(predictions, intents_json):
        """Provide an Answer to user's Questions.

        Provide the predictions and original data and a response from the original data would be randomly provided.

        Keyword arguments:
            predictions -- (list)
            data -- (dict)

        Returns:
            A dictionary that contains the JSON structured data.
        """

        tag = predictions[0]
        list_of_predictions = intents_json["intents"]
        for i in list_of_predictions:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
        return result