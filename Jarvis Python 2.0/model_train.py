import json
import pickle
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

with open("intents.json") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels=[]
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

number_of_classes = len(labels)

print(number_of_classes)

label_encoder =LabelEncoder()
label_encoder.fit(training_labels)
training_labels = label_encoder.transform(training_labels)

vocab_size = 1000
max_len = 20
ovv_token = "<OOV>"
embedding_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token=ovv_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(number_of_classes, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

model.summary()

history = model.fit(padded_sequences, np.array(training_labels), epochs = 1000)

model.save("chat_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file, protocol=pickle.HIGHEST_PROTOCOL)