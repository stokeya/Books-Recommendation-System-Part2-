#Import Libraries
import pandas as pd
from pathlib import Path
import numpy as np
import re

# Paths to original data
data_path = Path('Books_Reviews/books_data.csv')
ratings_path = Path('Books_Reviews/Books_rating.csv')

# Setup panda data frames
data_df = pd.read_csv(data_path, sep=',', on_bad_lines='warn', encoding='latin-1')
ratings_df = pd.read_csv(ratings_path, sep=',', on_bad_lines='warn', encoding='latin-1')

# Check for null values
data_df.isnull().sum()
ratings_df.isnull().sum()

# Remove null values from titles and reviews
data_df.dropna(subset=['Title'], inplace=True)

ratings_df.dropna(subset=['Title'], inplace=True)
ratings_df.dropna(subset=['review/text'], inplace=True)

# Remove unnescesary data columns
data_df.drop('image', axis=1, inplace=True)
data_df.drop('previewLink', axis=1, inplace=True)
data_df.drop('publisher', axis=1, inplace=True)
data_df.drop('publishedDate', axis=1, inplace=True)
data_df.drop('infoLink', axis=1, inplace=True)
data_df.drop('ratingsCount', axis=1, inplace=True)

ratings_df.drop('Price', axis=1, inplace=True)
ratings_df.drop('profileName', axis=1, inplace=True)
ratings_df.drop('review/time', axis=1, inplace=True)

# Remove duplicates from frame
ratings_df = ratings_df.drop_duplicates(['review/text'])

# Merge the datasets
merged_df = pd.merge(ratings_df, data_df)

# Create the csv using the merged dataset
merged_df.to_csv('Resources/complete_cleaned_data.csv')