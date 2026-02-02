import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(w for w in text.split() if w not in stop_words and len(w) > 2)
