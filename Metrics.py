import pandas as pd
import numpy as np
import mlflow.keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore

# --- 1. SETUP DATA ---
print("Preparing data for evaluation...")
df = pd.read_csv('Dataset/master_dataset.csv').dropna()
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

max_words = 20000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])

X_seq = pad_sequences(tokenizer.texts_to_sequences(df['text']), maxlen=max_len)
y = df['label_encoded'].values

# Split
val_idx = int(len(X_seq) * 0.8)
X_val = X_seq[val_idx:]
y_val = y[val_idx:]

# --- 2. LOAD MODEL FROM MLFLOW ---
RUN_ID = "8102a5fc3958413bbbdbbd62db0b6a37"
model_uri = f"runs:/{RUN_ID}/bilstm_glove_model"

print(f"Loading model from: {model_uri}")
try:
    model = mlflow.keras.load_model(model_uri)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Standard load failed, attempting direct path... Error: {e}")
    direct_path = rf"D:\SentimentAnalysis\mlruns\6\{RUN_ID}\artifacts\bilstm_glove_model"
    model = mlflow.keras.load_model(direct_path)

# --- 3. GENERATE PREDICTIONS & METRICS ---
print("Generating predictions...")
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate Text Report
print("\n" + "="*40)
print("CLASSIFICATION REPORT")
print("="*40)
# This report helps identify which emotions have low recall or precision
print(classification_report(y_val, y_pred, target_names=le.classes_))

# --- 4. PLOT CONFUSION MATRIX ---
plt.figure(figsize=(12,10))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Emotion Classification Confusion Matrix')
plt.show()