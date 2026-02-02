# model.py: Script to train and save the fake news detection model

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# Download NLTK resources if not already present
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess the text: lowercase, remove punctuation, lemmatize words.
    Keep words longer than 1 character, don't strip all stopwords.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 1]
    return " ".join(tokens)

print("Loading datasets...")
# Load FAKE news (label = 0)
fake_df = pd.read_csv('data/fake_news.csv')
fake_df['label'] = 0
print(f"Fake news loaded: {len(fake_df)} articles")

# Load REAL news (label = 1)
true_df = pd.read_csv('data/true_news.csv')
true_df['label'] = 1
print(f"Real news loaded: {len(true_df)} articles")

# Combine both datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
print(f"Total articles: {len(df)}")

# Combine title + text into one feature if available
if 'title' in df.columns and 'text' in df.columns:
    df['combined'] = df['title'].astype(str) + " " + df['text'].astype(str)
    df = df[['combined', 'label']].rename(columns={'combined': 'text'})
else:
    # fallback to text column only
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    df = df[[text_col, 'label']].rename(columns={text_col: 'text'})

# Remove empty/missing text
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != '']
print(f"After cleaning: {len(df)} articles")
print("Label distribution:")
print(df['label'].value_counts())

# Preprocess the text column
print("Preprocessing text...")
df['text'] = df['text'].apply(preprocess_text)

# Remove any empty texts after preprocessing
df = df[df['text'].str.strip() != '']
print(f"Final dataset size: {len(df)} articles")

# Split data (stratify to maintain class balance)
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# Create pipeline: TF-IDF + Logistic Regression with balanced weights
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)),
    ('model', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

# Train the model
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f'Confusion Matrix: TP={((y_pred==1) & (y_test==1)).sum()}, '
      f'TN={((y_pred==0) & (y_test==0)).sum()}, '
      f'FP={((y_pred==1) & (y_test==0)).sum()}, '
      f'FN={((y_pred==0) & (y_test==1)).sum()}')

# Save the pipeline (includes vectorizer and model)
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/fake_news_model.pkl')
print('\n🎉 Model saved to models/fake_news_model.pkl')
print('✅ Ready to run app.py!')