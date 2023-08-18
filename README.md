# Text-generation
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Step 1: Load and preprocess your text data
# Assume you have a text corpus as a list of sentences
corpus = [...]

# Step 2: Tokenize and prepare sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Create predictors and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# Step 3: Build an LSTM model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
y_categorical = np.array([np.eye(total_words)[word] for word in y])
model.fit(X, y_categorical, epochs=100, verbose=1)

# Step 5: Generate text
seed_text = "Once upon a time"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted_word_index = np.argmax(model.predict(token_list), axis=-1)
    predicted_word = tokenizer.index_word[predicted_word_index[0]]
    seed_text += " " + predicted_word

print(seed_text)
