import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 
import os
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tokenizer = tf.keras.preprocessing.text.Tokenizer()

text_data = open('test_data_copy.txt').read()
#print(text_data)

corpus = text_data.lower().split("\n")
#print(corpus)
tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index
total_words = len(tokenizer.word_index)+1
#print(word_index)
#print(total_words)


# create empty list for "xs"
in_sequences = []



for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        in_sequences.append(n_gram)


#print(in_sequences)

#create padding seq's
max_seq_len = max(len(x) for x in in_sequences)
#print(max_seq_len)

# pad sequences with a pre method. 
in_sequences = np.array(pad_sequences(in_sequences, maxlen = max_seq_len, padding='pre'))


#assign x's and labels
#get input of first part of sentence and label is what is supossed to come next in the seq
xs, labels = in_sequences[:,:-1],in_sequences[:,-1]

#create matrix for ys
ys = tf.keras.utils.to_categorical(labels,num_classes=total_words)

#testing
'''
print(in_sequences)
print(tokenizer.word_index['centre'])
print(tokenizer.word_index['my'])
print(xs[7])
print(ys[7])
print(tokenizer.word_index)
'''

#begin building the model for training
model = Sequential()
#subtract 1 due to the last word being chopped off to create a label
model.add(Embedding(total_words,240,input_length = max_seq_len - 1))
model.add(Bidirectional(LSTM(150)))
# use softmax for probability vector vs defualt linear
model.add(Dense(total_words,activation='softmax'))
# create adam algo
adam = Adam(learning_rate=0.01)
#use categorical_crossentropy for labels to predictions comparision with metrics focused on accuracy
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

history = model.fit(xs, ys, epochs=100, verbose=1)

#test
print(model)

#graph model
'''
def graph_model(history,string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

graph_model(history,'accuracy')
'''

#produce text
seed_text = "Thou shall not pass this test"
next_words = 200

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
	predicted = np.argmax(model.predict(token_list), axis=-1)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)







