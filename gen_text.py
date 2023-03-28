import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 
import matplotlib.pyplot as plt
import string


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tokenizer = tf.keras.preprocessing.text.Tokenizer()

text_data = open('train_data.txt').read()



#Clean the text
#text_data = re.sub('[^A-Za-z0-9]+', ' ', text_data)
text_data = text_data.translate(str.maketrans('','', string.punctuation))
#â€™
text_data = text_data.replace('â',"")
text_data = text_data.replace('€',"")
text_data = text_data.replace('™',"")



corpus = text_data.lower().split("\n")
#print(corpus)
tokenizer.fit_on_texts(corpus)

#word_index = tokenizer.word_index
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

#testing can remove comments
'''
print(text_data)
print()
print(tokenizer.word_index)
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
model.add(LSTM(200))
# use softmax for probability vector vs defualt linear
model.add(Dense(total_words,activation='softmax'))
# create adam algo
adam = Adam(learning_rate=0.01)
#use categorical_crossentropy for labels to predictions comparision with metrics focused on accuracy
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print("-----------------------------------------")
print("Sonnet bot is thinking....")
print('-----------------------------------------')
history = model.fit(xs, ys, epochs=5, verbose=1)

#saves the model for later use





#graph model

def graph_model(history,string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
graph_model(history,'accuracy')


#produce text
seed_text = input("Enter start of sonnet: ")
next_words = 150
#print(tokenizer.word_index.items())

print("-----------------------------------------")
print("AI generated sonnet based on user inputed seed.")
print('-----------------------------------------')
for i in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
	predicted = np.argmax(model.predict(token_list,verbose=0), axis=-1)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)

