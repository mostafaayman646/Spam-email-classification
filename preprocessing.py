import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
   
    text = str(text).lower()                 
    text = re.sub(r'\W', ' ', text)          
    text = re.sub(r'\d+', '', text)     
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return " ".join(filtered)

def lemmatize_text(text):

    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)

def preprocess_text(text):

    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\hom\Downloads\spam.csv", encoding='latin-1')

    df = df[['class', 'message']]
    df.columns = ['label', 'text']

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df['clean_text'] = df['text'].apply(preprocess_text)

    print("\n✅ first five after preprocessing")
    print(df[['label', 'text', 'clean_text']].head())

    df.to_csv(r"C:\Users\hom\Downloads\spam_cleaned.csv", index=False, encoding='utf-8')
    print("\n💾 spam cleaned saved")
