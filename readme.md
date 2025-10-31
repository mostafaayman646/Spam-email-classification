# Spam Email Classification

## Folder Structure

```
Spam-email-classification/
├── Data_helper.py           # Functions to load and save data
├── preprocessing.py         # Text cleaning, stopword removal, lemmatization
├── model_utils.py           # Model saving utility
├── train.py                 # End-to-end training script
├── train.ipynb              # Step-by-step notebook for training and evaluation
├── requirements.txt         # Python dependencies
├── readme.md                # Project documentation
├── Data/
│   ├── spam.csv             # Raw dataset
│   └── spam_cleaned.csv     # Cleaned dataset
├── Model/
│   └── XGBoost_pipeline.pkl # Saved model pipeline
└── Figure_1.png             # Example output figure
```

## Steps Performed in Each File

- **Data_helper.py**: Loads raw and cleaned data, saves processed data.
- **preprocessing.py**: Cleans text, removes stopwords, lemmatizes, preprocesses dataset, saves cleaned data.
- **model_utils.py**: Saves the trained pipeline using joblib.
- **train.py**: Loads cleaned data, splits into train/test, builds pipeline, trains XGBoost, evaluates, saves model, plots confusion matrix.
- **train.ipynb**: Interactive notebook version of training and evaluation steps.

## How to Use This Repository

1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
2. **Preprocess the data**:
   ```powershell
   python preprocessing.py
   ```
   - This will create `Data/spam_cleaned.csv`.
3. **Train the model**:
   ```powershell
   python train.py
   ```
   - This will train the XGBoost model and save it to `Model/XGBoost_pipeline.pkl`.
   - It will also display accuracy and confusion matrix.
4. **Use the notebook**:
   - Open `train.ipynb` in VS Code or Jupyter for step-by-step execution and visualization.

## Final Results (XGBoost)

- **Training Accuracy**: 0.9845
- **Test Accuracy**: 0.9748

The XGBoost model achieves high accuracy on both training and test sets, indicating strong performance for spam email classification.