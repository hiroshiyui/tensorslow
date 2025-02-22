#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import random

# 1. Generate Random Sentences (or use your existing data)

def generate_random_sentence(vocabulary, max_length):
    sentence_length = random.randint(1, max_length)
    sentence = [random.choice(list(vocabulary.keys())) for _ in range(sentence_length)]
    return " ".join(sentence)

def create_random_sentences(num_sentences, vocabulary, max_length):
    sentences = [generate_random_sentence(vocabulary, max_length) for _ in range(num_sentences)]
    return sentences

# Example Vocabulary (REPLACE with your vocabulary from real data)
vocabulary = {  # Example - replace with your real vocabulary
    "the": 1, "quick": 2, "brown": 3, "fox": 4, "jumps": 5, "over": 6, "lazy": 7, "dog": 8,
    "cat": 9, "sleeps": 10, "eats": 11, "runs": 12, "happy": 13, "sad": 14, "big": 15, "small": 16,
    "sentence": 17, "word": 18, "example": 19, "data": 20, "model": 21, "network": 22, "learning": 23
}  # Expanded example vocabulary

num_sentences = 100000  # Adjust as needed
max_length = 20  # Adjust as needed
sentences = create_random_sentences(num_sentences, vocabulary, max_length)


# Generate Random Labels (REPLACE with your real labels)
labels = np.random.randint(0, 2, size=num_sentences)  # 0 or 1 (example)



# 2. Tokenize and Pad (using the same tokenizer for training and new data)

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(sentences)  # Fit on training data
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')



# 3. Build, Compile, and Train the Model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
    tf.keras.layers.LSTM(32), # LSTM is usually a good choice
    tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(padded_sequences, labels, epochs=10, batch_size=32)  # Adjust epochs and batch size


# 4. Evaluate the Model (Optional - requires splitting data into train/test)
# ... (Code to split data and evaluate) ...

# 5. Make Predictions on New Data

def predict_on_new_sentences(new_sentences, tokenizer, max_length, model):
    new_sequences = tokenizer.texts_to_sequences(new_sentences)
    new_padded = tf.keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=max_length, padding='post')
    predictions = model.predict(new_padded)
    return predictions


new_sentences = [
    "the quick brown fox jumps over the lazy dog",
    "the cat sleeps",
    "a new sentence example",
    "this is a test",
    "another example sentence"
]

new_sentences_random = create_random_sentences(5, vocabulary, max_length)

all_new_sentences = new_sentences + new_sentences_random # Combine both types of sentences

predictions = predict_on_new_sentences(all_new_sentences, tokenizer, max_length, model)
print("Predictions on new sentences:")
print(predictions)
