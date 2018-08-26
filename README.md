# Project: Disaster Response Pipeline

#### A pipeline that analyzes and categorizes messages sent during disaster events.  The project consists of an ETL and ML pipeline and a Flask App to present the findings.  

#### The ETL pipeline can be found in the `process_data.py` file and it handles the following tasks:

* Combines the two given datasets
* Cleans the data
* Stores it in a SQLite database

#### The ML pipeline can be found in the `train_classifier.py` file and it handles the following tasks:

* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports your final model as a pickle file