import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# --- 1. DATA PREPARATION (Defines the missing variables) ---
print("Loading data for fine-tuning...")
df = pd.read_csv('Dataset/master_dataset.csv').dropna()

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)

# Tokenize text (Ensure parameters match your original training)
max_words = 20000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])

X_seq = pad_sequences(tokenizer.texts_to_sequences(df['text']), maxlen=max_len)
y = df['label_encoded'].values

# Split data - This defines y_train and X_train_seq
X_train_seq, X_val_seq, y_train, y_val = train_test_split(X_seq, y, test_size=0.2, stratify=y)

# --- 2. LOAD PREVIOUS MODEL ---
RUN_ID = "8102a5fc3958413bbbdbbd62db0b6a37"
model_uri = f"runs:/{RUN_ID}/bilstm_glove_model"
model = mlflow.keras.load_model(model_uri)

# --- 3. CALCULATE CLASS WEIGHTS ---
# Now y_train is defined and can be used here
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(weights))

# --- 4. UNFREEZE & FINE-TUNE ---
model.layers[0].trainable = True # Unfreeze GloVe layer
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mlflow.set_experiment("Emotion_BiLSTM_FineTuning")
with mlflow.start_run(run_name="FineTuning_Unfrozen_GloVe"):
    history = model.fit(
        X_train_seq, y_train, 
        epochs=5, 
        batch_size=64, 
        validation_data=(X_val_seq, y_val),
        class_weight=class_weight_dict # Apply weights to fix 'Joy' bias
    )
    mlflow.keras.log_model(model, "fine_tuned_bilstm_model")

print("Fine-tuning complete!")