import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Data_helper import *

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
    df = Load_data()
    
    df = df[['class', 'message']]
    df.columns = ['label', 'text']

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df['clean_text'] = df['text'].apply(preprocess_text)

    df = df.drop(columns=['text'])
    df = df.dropna()
    df = df.drop_duplicates()
    
    print("\n✅ first five after preprocessing")
    print(df[['label', 'clean_text']].head())

    Save_data(df)
    print("\n💾 spam cleaned saved")