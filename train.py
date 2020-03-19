import os
import re
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.models import Sequential


pospath = os.path.join(os.getcwd(),'train','pos')
negpath = os.path.join(os.getcwd(),'train','neg')

def get_array_from_directory(path):
    array = os.listdir(path)
    m = []
    for n in range(len(array)):
        f=open(os.path.join(path,array[n]), encoding="utf8")
        data = f.read()
        words = text_to_word_sequence(data)
        result = one_hot(data, round(len(words)*1.3))
        # print(result)
        m.append(result)

    return m


posarray = get_array_from_directory(pospath)
negarray = get_array_from_directory(negpath)

poslabelarr = np.zeros(12500)
neglabelarr = np.ones(12500)

for i in range(len(negarray)):
    posarray.append(negarray[i])


features = posarray
features = pad_sequences(features, padding='post', maxlen=2000)
# features = np.array(features).astype('f')
# features=np.concatenate((posarray,negarray))

labels = np.concatenate((poslabelarr,neglabelarr))
labels = keras.utils.to_categorical(labels,2)

# print(features)
# print(len(features))
# print(len(features[0]))
# print(labels)

model = Sequential()
# model.add(Embedding(2000,128))
# model.add(Dropout(0.4))
# model.add(LSTM(128))
# model.add(Dense(64))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

model.add(Embedding(2000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(features,labels,epochs=25)

model.save('sentimentmodel.h5')