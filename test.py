import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# filepath = 'D:/AI&ML/data/aclImdb_v1/aclImdb/train/unsup/0_0.txt'

data="I have never seen this kind of film, very bad story. all time i got bored"
# f = open(filepath, encoding='utf8')
# data=f.read()
words=text_to_word_sequence(data)
wordarr=one_hot(data, round(len(words)*1.3))
wordarr = np.array(wordarr).astype('f')
wordarr=np.expand_dims(wordarr,axis=0)
wordarr = pad_sequences(wordarr, maxlen=2000)

model = load_model('model.h5')

result = model.predict([wordarr])  #here wordarr pass as list so [wordarr]
print(result)

if result[0,0]>result[0,1]:
    print('positive sentiment')
elif result[0,0]<result[0,1]:
    print('negative sentiment')
elif result[0,0]==result[0,1]:
    print('neutral sentiment')



