"""
ML Pipeline that handles the following steps:

* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports your final model as a pickle file
"""
# import libraries
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

def load_data_from_db():
    # load data from database
    engine = create_engine('sqlite:///df.db')
    df = pd.read_sql_query('SELECT * FROM df', con = engine).drop('index', axis = 1)
    X = df.loc[:, :].drop(['id', 'genre', 'original'], axis = 1)
    X = df.loc[:, 'message']
    y = df.loc[:, 'genre']
    return X, y  

def partition_data(X, y):
    # Partition the data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Reshape labels
    y_train = np.reshape(y_train.values, (y_train.shape[0], 1))
    y_test = np.reshape(y_test.values, (y_test.shape[0], 1))
    return X_train, X_test, y_train, y_test

def tokenize(text):
    text = re.sub(r"[^A-Za-z]", " ", text)  
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model(clf, parameters = None, apply_grid_search = False):
    """Builds an ML pipeline
    
    Args:
        clf: Estimator
        param_grid: dict
        apply_grid_search: bool
            Turns on/off GridSearchCV
        
    Return: 
        model
    """
    model = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', clf)])
    
    # Perform GridSearchCV if switch is ON
    if apply_grid_search: 
        model = GridSearchCV(model, param_grid = parameters)
    return model


def main():
    X, y = load_data_from_db()
    X_train, X_test, y_train, y_test = partition_data(X, y)

    # Build model
    clf = MultiOutputClassifier(RandomForestClassifier())
    parameters = {'clf__estimator__n_estimators': [15, 20]}
    # GridSearchCV turned off.  Set apply_grid_search attribute to True to turn it on
    # Also, remember to pass in the parameters dictionary
    model = build_model(clf, parameters = parameters) 
    # Train model
    model.fit(X_train, y_train)

    # Label names
    labels = y.unique()
    # Predictions
    y_pred = model.predict(X_test)

    # Show results
    print(classification_report(y_test, y_pred, target_names = labels)) 

    # Export model as a pickle file
    pickle.dump(clf, open('model', "wb"))

if __name__ == "__main__":

    main()
