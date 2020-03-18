import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

filepath = 'D:\\AI&ML\\imdb sentiment analysis\\train\pos\\0_9.txt'

f = open(filepath, encoding='utf8')
data=f.read()
words=text_to_word_sequence(data)
wordarr=one_hot(data, round(len(words)*1.3))
wordarr = np.array(wordarr).astype('f')
wordarr=np.expand_dims(wordarr,axis=0)
wordarr = pad_sequences(wordarr, padding='post', maxlen=2000)

model = load_model('sentimentmodel.h5')

result = model.predict(wordarr)
print(result)