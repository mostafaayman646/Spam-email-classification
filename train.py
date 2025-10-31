import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns

import pickle

from Data_helper import *
from model_utils import *

if __name__ == '__main__':
    RANDOM_STATE = 42
    
    df = Load_data('Data/spam_cleaned.csv')
    df = df.dropna()
    df = df.drop_duplicates()
    
    X = df['clean_text']
    Y = df['label']

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=RANDOM_STATE)

    pipe = Pipeline([
        ('bow', CountVectorizer()),
        ('model', xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=7,
            n_estimators=80,
            use_label_encoder=False,
            eval_metric='auc',
        ))
    ])

    pipe.fit(x_train, y_train)

    y_pred_class = pipe.predict(x_test)
    y_pred_train = pipe.predict(x_train)

    print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
    print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

    cm = metrics.confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Ham (0)", "Spam (1)"],
                yticklabels=["Ham (0)", "Spam (1)"])
    plt.title("XGBoost Confusion Matrix (Test Set)", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    # save_model(pipe)