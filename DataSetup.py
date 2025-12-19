import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#", "", text) 
    text = re.sub(r"[^\w\s]", "", text) 
    return text.strip()

mapping = {
    'admiration': 'joy', 'amusement': 'joy', 'approval': 'joy', 'excitement': 'joy', 
    'gratitude': 'joy', 'joy': 'joy', 'love': 'joy', 'optimism': 'joy', 'pride': 'joy',
    'relief': 'joy', 'grief': 'sadness', 'remorse': 'sadness', 'sadness': 'sadness',
    'disappointment': 'sadness', 'anger': 'anger', 'annoyance': 'anger', 
    'disapproval': 'anger', 'nervousness': 'fear', 'fear': 'fear',
    'surprise': 'surprise', 'realization': 'surprise', 'curiosity': 'surprise',
    'neutral': 'neutral', 'happiness': 'joy', 'worry': 'sadness', 'hate': 'anger',
    'shame': 'sadness', 'guilt': 'sadness', 'disgust': 'disgust'
}

def load_and_standardize():
    print("--- Starting Data Setup ---")

    # 1. Twitter
    tw = pd.read_csv('Dataset/twitter.csv', encoding='latin-1')
    tw = tw[['text', 'sentiment']].rename(columns={'sentiment': 'label'})

    # 2. ISEAR
    isear = pd.read_csv('Dataset/isear.csv', sep='|', on_bad_lines='skip')
    isear = isear.iloc[:, [0, -1]]
    isear.columns = ['label', 'text']

    # 3. GoEmotions (The One-Hot Fix)
    print("Processing GoEmotions (One-Hot Encoded)...")
    g1 = pd.read_csv('Dataset/goemotions_1.csv')
    g2 = pd.read_csv('Dataset/goemotions_2.csv')
    g3 = pd.read_csv('Dataset/goemotions_3.csv')
    go_raw = pd.concat([g1, g2, g3], ignore_index=True)
    
    # Identify the emotion columns (they are the ones in our mapping)
    emotion_cols = [col for col in go_raw.columns if col in mapping.keys()]
    
    # idxmax(axis=1) finds the column name with the highest value (the 1)
    go_raw['label'] = go_raw[emotion_cols].idxmax(axis=1)
    go = go_raw[['text', 'label']]

    # Merge All
    print("Merging and Mapping...")
    master_df = pd.concat([tw, isear, go], ignore_index=True)
    master_df['label'] = master_df['label'].map(mapping)
    
    # Cleaning
    master_df = master_df.dropna(subset=['label', 'text'])
    master_df['text'] = master_df['text'].apply(clean_text)
    master_df = master_df[master_df['text'] != ""]
    
    return master_df

if __name__ == "__main__":
    df = load_and_standardize()
    print("\n--- Final Dataset Statistics ---")
    print(df['label'].value_counts())
    df.to_csv('Dataset/master_dataset.csv', index=False)
    print("\nSuccess! Combined dataset saved.")