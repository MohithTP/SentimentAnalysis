import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore 
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout #type:ignore

# Load and Split Data
df = pd.read_csv('Dataset/master_dataset.csv').dropna()
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

num_classes = len(le.classes_) 
print(f"Total unique emotions: {num_classes}")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['text'], df['label_encoded'], test_size=0.2, stratify=df['label_encoded']
)

mlflow.set_experiment("Emotion_Model_Comparison")

# --- MODEL 1: Logistic Regression ---
with mlflow.start_run(run_name="Logistic_Regression"):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    
    # Log Metrics
    train_acc = lr.score(X_train_tfidf, y_train)
    test_acc = lr.score(X_test_tfidf, y_test)
    mlflow.log_metrics({"train_acc": train_acc, "test_acc": test_acc})
    mlflow.sklearn.log_model(lr, "model")

# --- Prep for DL (LSTM & GRU) ---
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train_raw)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=100)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=100)

def train_dl(name, layer):
    with mlflow.start_run(run_name=name):
        model = Sequential([
            Embedding(10000, 128, input_length=100),
            layer(64),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train and Log
        history = model.fit(X_train_seq, y_train, epochs=5, validation_data=(X_test_seq, y_test))
        
        # Log final metrics
        mlflow.log_metrics({
            "train_acc": history.history['accuracy'][-1],
            "test_acc": history.history['val_accuracy'][-1]
        })
        mlflow.keras.log_model(model, "model")

train_dl("LSTM_Model", LSTM)
train_dl("GRU_Model", GRU)