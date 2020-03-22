import os
import re
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.models import Sequential


def get_array_from_directory(path):
    array = os.listdir(path)
    m = []
    for n in range(len(array)):

        with open(os.path.join(path, array[n]), encoding='utf8') as f:
            data = f.read()
            words = set(text_to_word_sequence(data))
            result = one_hot(data, round(len(words) * 1.3))
            m.append(result)

    m = pad_sequences(m, maxlen=2000)
    return m

posarray = get_array_from_directory('D:/AI&ML/data/aclImdb_v1/aclImdb/train/pos')
negarray = get_array_from_directory('D:/AI&ML/data/aclImdb_v1/aclImdb/train/neg')

poslabelarr = np.zeros(len(posarray))
neglabelarr = np.ones(len(negarray))

x_train = np.concatenate((posarray, negarray))

y_train = np.concatenate((poslabelarr, neglabelarr))
y_train = keras.utils.to_categorical(y_train, 2)

model = Sequential()
model.add(Embedding(2000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=25)
model.save('model.h5')
