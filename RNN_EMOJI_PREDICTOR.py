import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import emoji
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


data = pd.read_csv('/Users/debar/Desktop/python/RNN/emoji_text.csv', header=None)
#emoji mapping to number
import emoji

emoji_dict = {
    0: ":red_heart:",
    1: ":baseball:",
    2: ":grinning_face_with_big_eyes:",
    3: ":disappointed_face:",
    4: ":fork_and_knife_with_plate:"
}

# This function will convert the answer of RNN into an emoji
def label_to_emoji(label):
    return emoji.emojize(emoji_dict[label])

# Example usage
print(label_to_emoji(0))  # Outputs ❤️ if emoji package supports it

# X = feature, Y = label
X = data[0].values
Y = data[1].values

file = open('/Users/debar/Desktop/python/RNN/glove.6B.100d.txt', 'r', encoding='utf8')
content = file.readlines()
file.close()  # Ensure parentheses here to actually call the function
#print(content)
embedding = {}
for line in content[:]:
    line = line.split()
    embedding[line[0]] = np.array(line[1:],dtype=float)
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
word2line = tokenizer.word_index  # This is a dictionary mapping words to their integer indices

Xtokens = tokenizer.texts_to_sequences(X)

def get_maxlen(data):
    maxLen = 0;
    for word in data:
        maxLen = max(maxLen,len(word))
    return maxLen

maxlen = get_maxlen(Xtokens)
Xtrain = pad_sequences(Xtokens,maxlen = maxlen, padding = 'post',truncating = 'post')
#does padding in suffix
embedding_marix = np.zeros((len(word2line)+1,100))

for word, i in word2line.items():
    embedding_vector = embedding[word]
    embedding_marix[i] = embedding_vector
    
Ytrain = to_categorical(Y)
print(Ytrain)

model = keras.models.Sequential([
    keras.layers.Embedding(input_dim = len(word2line)+1,
                           output_dim = 100,
                           weights = [embedding_marix],
                           trainable = False
                           ),
    keras.layers.LSTM(units = 16, return_sequences = True),
    keras.layers.LSTM(units = 4),
    keras.layers.Dense(5, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(
    Xtrain,
    Ytrain,
    epochs = 500
)

test = ["i feel good", "i love my mom", "lets play baseball"]
test_seq = tokenizer.texts_to_sequences(test)
Xtest = pad_sequences(test_seq, maxlen =maxlen, padding = 'post', truncating = 'post')

Y_pred = model.predict(Xtest)
print(Y_pred)
Y_pred = np.argmax(Y_pred,axis = 1)
for i in range(len(test)):
    print(test[i], label_to_emoji(Y_pred[i]))
