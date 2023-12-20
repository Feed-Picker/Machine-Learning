import numpy as np
import pandas as pd
import string
import pickle  # Import the pickle module
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
data = pd.read_csv("fix_dataset_SA.csv")

# Check for missing values
data = data.dropna(subset=['Review', 'Hasil Perbandingan'])

# Extract relevant columns
mpermanent = data[['Review', 'Hasil Perbandingan']]

# Process the text data
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Apply text processing to the 'Review' column
mpermanent['clean_review'] = mpermanent['Review'].apply(text_process)

# Join the processed words into sentences
mpermanent['clean_sentences'] = mpermanent['clean_review'].apply(lambda x: ' '.join(x))

# Keras Tokenizer with a specified vocabulary size
vocab_size = 1000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(mpermanent['clean_sentences'])
text_sequences = tokenizer.texts_to_sequences(mpermanent['clean_sentences'])

# Save the tokenizer to a pickle file
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Encode labels
labels = mpermanent['Hasil Perbandingan']
label_mapping = {'Positive': 2, 'Negative': 0, 'Neutral': 1}
Y = np.array([label_mapping[label] for label in labels])

# Pad sequences
max_cap = 20
X = pad_sequences(text_sequences, maxlen=max_cap)

# Convert labels to one-hot encoding
Y_one_hot = pd.get_dummies(labels).values

# Split the data into training and testing sets
split_ratio = 0.7
split_index = int(split_ratio * len(X))
X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], Y_one_hot[:split_index], Y_one_hot[split_index:]

# Build the LSTM model with increased dropout and bidirectional LSTM
embedding_dim = 100
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_cap))
model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))
model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Save the best model automatically
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model with early stopping and model checkpoint
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])

# Evaluate the model (no need to load the best model explicitly)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert one-hot encoded y_test back to labels
y_test_labels = np.argmax(y_test, axis=1)

# Simplify the mapping of labels
label_mapping_reverse = {v: k for k, v in label_mapping.items()}
results_df = pd.DataFrame({'Review': mpermanent['Review'].iloc[split_index:], 'Actual_Label': y_test_labels, 'Predicted_Label': y_pred})
results_df['Actual_Label'] = results_df['Actual_Label'].map(label_mapping_reverse)
results_df['Predicted_Label'] = results_df['Predicted_Label'].map(label_mapping_reverse)


# Save actual vs predicted results to a CSV file
results_df.to_csv('actual_vs_predicted_results.csv', index=False)
print("Actual vs Predicted results saved to actual_vs_predicted_results.csv")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred))

# Display counts of actual classes and predicted classes
actual_counts = results_df['Actual_Label'].value_counts()
predicted_counts = results_df['Predicted_Label'].value_counts()

# Display sum of actual and predicted data based on its class
print("\nSum of Actual Data Based on Class:")
print(actual_counts)

print("\nSum of Predicted Data Based on Class:")
print(predicted_counts)

# Display final accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Final Accuracy: {accuracy * 100:.3f}%")

# Display confusion matrix
conf_mat = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training accuracy and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Plot training loss and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()