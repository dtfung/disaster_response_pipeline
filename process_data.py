"""
ETL Pipeline that takes any dataset and performs the following tasks:

* Combines the two given datasets
* Cleans the data
* Stores it in a SQLite database
"""
# import libraries
import sys
import pandas as pd
from settings import *
from sqlalchemy import create_engine

def retrieve_filename():
    """
    Accesses the filename that is passed in as a command line 
    argument when this file is run.  
    """
    return sys.argv[1]

def load_data(filename):
    """Load dataset
    Args:
        filename: str
            Contains location to data

    Return:
        df: Pandas DataFrame
    """
    df = pd.read_csv(filename)
    return df

def process_categorical_data(df):
    """Splits categories and converts categories to numbers

    Args:
        df: Pandas DataFrame

    Return 
        categories: Pandas DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # Use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: str(x).split('-')[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: str(x).split('-')[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    return categories

def process():
    messages = load_data(MESSAGES_FILE)
    categories = load_data(CATEGORIES_FILE)
    df = messages.merge(categories, on = 'id') # merge datasets
    categories = process_categorical_data(df)
    # drop the original categories column from `df`
    df.drop('categories', inplace = True, axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df[~df.duplicated()]
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///df.db')
    df.to_sql('df', engine)

if __name__ == "__main__":

    process()


