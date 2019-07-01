import numpy as np

from keras import backend as K
def custom_sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())

def input_preprocess(word, tokenizer, sequence_length):
    """
    Preprocess the input
    Return : Preprocessed input and the Tokenizer
    """
    preprocess_x = tokenizer.texts_to_sequences(word)

    result = [0 for i in range(sequence_length)]

    k = 0

    for i in range(len(preprocess_x)):
        try :
            result[k] = preprocess_x[i][0]
            k = k + 1
        except :
            pass

    return np.array([result]).astype('int32')

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ' '

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def get_prediction(bengali_word, bengali_tokenizer, max_seq_length, english_tokenizer):
    p_word = pre_processed_bengali_word(bengali_word)
    preproc_input = input_preprocess(p_word, bengali_tokenizer, max_seq_length)
    prediction = logits_to_text(model.predict(preproc_input)[0], english_tokenizer)

    result = ""
    for i in prediction.strip():
        if i != ' ':
            result = result + i.upper()

    return result

def pre_processed_bengali_word(bengali_word):
    result = ""
    for i in bengali_word:
        result = result + i + " "
    return result

######################################################################################
# Loading of necessary files

from keras.models import load_model
model = load_model('transliteration_model.h5', custom_objects={'custom_sparse_categorical_accuracy': custom_sparse_categorical_accuracy})
import pickle

with open("ben_token.pickle", "rb") as fp:
    bengali_tokenizer = pickle.load(fp)

with open("eng_token.pickle", "rb") as fp:
    english_tokenizer = pickle.load(fp)

with open("max_seq_length.txt", "rb") as fp:
    max_bengali_seq_length = int(fp.read())

######################################################################################
