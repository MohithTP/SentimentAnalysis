import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# --- 1. Load Data ---
df = pd.read_csv('Dataset/master_dataset.csv').dropna()
X = df['text']
y = pd.get_dummies(df['label']).values
num_classes = y.shape[1]

# Tokenization
max_words = 20000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_len)

# --- 2. Load GloVe Embeddings ---
print("Loading GloVe Embeddings...")
embeddings_index = {}
with open('Dataset/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')

# Create Embedding Matrix
embedding_matrix = np.zeros((max_words, 100))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# --- 3. Build Bidirectional Model ---
def build_bilstm():
    model = Sequential([
        # Load GloVe weights into Embedding layer
        Embedding(max_words, 100, weights=[embedding_matrix], trainable=False),
        
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 4. MLflow Experiment ---
mlflow.set_experiment("Emotion_BiLSTM_GloVe")

with mlflow.start_run(run_name="BiLSTM_with_GloVe_v1"):
    model = build_bilstm()
    history = model.fit(X_seq, y, epochs=10, batch_size=64, validation_split=0.2)
    
    # Log metrics
    mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
    mlflow.keras.log_model(model, "bilstm_glove_model")
    print(f"Training Complete. Val Accuracy: {history.history['val_accuracy'][-1]}")