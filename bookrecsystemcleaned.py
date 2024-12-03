# -*- coding: utf-8 -*-
"""BookRecSystem.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VGzsWERvY6YlTFngXr9bAy_TvKyPceqU

Create users and books files
"""

# Takes all unique books from ratings_cleaned (2).csv (all the ones with a review) and adds them to users_cleaned (2).csv

import pandas as pd

# Gets unique users and adds them to data set

ratings_df = pd.read_csv('complete_cleaned_data.csv')

unique_users = pd.DataFrame(ratings_df['User_id'].unique(), columns=['User_id'])

unique_users.reset_index(inplace=True)

unique_users.to_csv('users_from_complete_cleaned_data.csv', index=False)

# Gets unique books and adds them to data set

ratings_df = pd.read_csv('complete_cleaned_data.csv')

unique_users = pd.DataFrame(ratings_df['Id'].unique(), columns=['Id'])

unique_users.reset_index(inplace=True)

unique_users.to_csv('books_from_complete_cleaned_data.csv', index=False)

"""Add sentiment analysis to file"""

import pandas as pd
from textblob import TextBlob


# Calculate sentiment score
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    scaled_score = (polarity + 1) * 2 + 1
    return round(scaled_score)


df = pd.read_csv('complete_cleaned_data.csv', engine='python')

columns = list(df.columns) + ['sentiment_score']

with open('with_rating_complete_cleaned_data.csv', 'a', newline='', encoding='utf-8') as f:
    f.write(','.join(columns) + '\n')

# Loop through each review (Can take a while to run code)
for i in range(len(df)):
    review_text = df.loc[i, 'review/text']
    user_rating = df.loc[i, 'review/score']
    user_helpfulness = df.loc[i, 'review/helpfulness']

    numerator, denominator = user_helpfulness.split("/")
    if float(denominator) == 0:
        result = 1
    else:
        result = float(numerator) / float(denominator)

    sentiment_score = analyze_sentiment(review_text) + int(user_rating * result)

    row_data = list(df.loc[i]) + [sentiment_score]

    # Calculate scores and add them to final dataset
    with open('with_rating_complete_cleaned_data.csv', 'a', newline='', encoding='utf-8') as f:
        f.write(','.join(map(str, row_data)) + '\n')

"""Main portion of project:"""

# Imports
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load files
books_path = Path('books_from_complete_cleaned_data.csv')
ratings_path = Path('with_rating_complete_cleaned_data4.csv')
users_path = Path('users_from_complete_cleaned_data.csv')


# Handle lines with errors
def handle_bad_lines(error):
    return None


books_df = pd.read_csv(books_path, index_col=0)
ratings_df = pd.read_csv(ratings_path, index_col=0, on_bad_lines=handle_bad_lines, engine='python')
users_df = pd.read_csv(users_path, index_col=0)

# Reformat files
ratings_df.columns = ['Id', 'Title', 'User_id', 'review/helpfulness', 'review/score', 'review/summary', 'review/text',
                      'description', 'authors', 'categories', 'sentiment_score']
print(ratings_df)

# Manage data sets to new merged data frame
user_ratings_df = pd.merge(users_df, ratings_df, on='User_id', how='inner')
books_user_ratings_df = pd.merge(user_ratings_df, books_df, on='Id', how='inner')

# Sort books
sorted_book_ratings = books_user_ratings_df.groupby('Title')['sentiment_score'].mean().sort_values(ascending=False)

# Manipulate ratings data
ratings_data = pd.DataFrame(books_user_ratings_df.groupby('Title')['sentiment_score'].mean())

ratings_data['Ratings-Count'] = pd.DataFrame(books_user_ratings_df.groupby('Title')['sentiment_score'].count())

ratings_data.rename(columns={'sentiment_score': 'Average-Rating'}, inplace=True)

# Group items
user_freq = books_user_ratings_df[['User_id', 'Title']].groupby('User_id').count().reset_index()
user_freq.columns = ['User_id', 'Ratings-Count']
user_freq.sort_values(by='Ratings-Count', ascending=False).head(10)

# Sort mean ratings
mean_rating = books_user_ratings_df[['sentiment_score', 'Title']].groupby('Title')[['sentiment_score']].mean()
mean_rating.sort_values(by='sentiment_score', ascending=False).head()

lowest_rated = mean_rating['sentiment_score'].idxmin()
books_user_ratings_df.loc[books_user_ratings_df['Title'] == lowest_rated]

highest_rated = mean_rating['sentiment_score'].idxmax()
books_user_ratings_df.loc[books_user_ratings_df['Title'] == highest_rated]

books_stats = books_user_ratings_df.groupby('Title')[['sentiment_score']].agg(['count', 'mean'])
books_stats.columns = books_stats.columns.droplevel()
books_stats.sort_values(by='count', ascending=False).head(10)

# Set book ids accordingly
book_ids = pd.unique(books_user_ratings_df['Id'].to_numpy())
book_ids = pd.Series(np.arange(len(book_ids)), book_ids)
book_ids = pd.DataFrame(book_ids)
book_ids.reset_index(inplace=True)
book_ids.rename(columns={'index': 'Id', 0: 'Book-ID'}, inplace=True)

books_user_ratings_df = pd.merge(books_user_ratings_df, book_ids, on='Id', how='left')


# Sets variables from data frame
def create_matrix(df):
    N = len(df['User_id'].unique())
    M = len(df['Book-ID'].unique())

    user_mapper = dict(zip(np.unique(df["User_id"]), list(range(N))))
    book_mapper = dict(zip(np.unique(df["Book-ID"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["User_id"])))
    book_inv_mapper = dict(zip(list(range(M)), np.unique(df["Book-ID"])))

    user_index = [user_mapper[i] for i in df['User_id']]
    book_index = [book_mapper[i] for i in df['Book-ID']]

    X = csr_matrix((df["Book-ID"], (book_index, user_index)), shape=(M, N))

    return X, user_mapper, book_mapper, user_inv_mapper, book_inv_mapper


books_user_ratings_df['User_id'] = books_user_ratings_df['User_id'].astype(str)
X, user_mapper, book_mapper, user_inv_mapper, book_inv_mapper = create_matrix(books_user_ratings_df)


# Finds similar books
def find_similar_books(book_id, X, books_user_ratings_df, k, metric='cosine', show_distance=False):
    X, user_mapper, book_mapper, user_inv_mapper, book_inv_mapper = create_matrix(books_user_ratings_df)

    neighbour_ids = []

    book_ind = book_mapper[book_id]
    book_vec = X[book_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    book_vec = book_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(book_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(book_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids


book_titles = dict(zip(books_user_ratings_df['Book-ID'], books_user_ratings_df['Title']))


# Get book ID from title
def get_book_id_from_title(title, books_user_ratings_df):
    try:
        book_id = \
        books_user_ratings_df.loc[books_user_ratings_df['Title'].str.lower() == title.lower(), 'Book-ID'].values[0]
        return book_id
    except IndexError:
        return None


# Get information from book to be printed
def get_book_details(title_name):
    book_data = ratings_df[ratings_df['Title'] == title_name]
    if book_data.empty:
        return

    description = book_data['description'].iloc[0]
    authors = book_data['authors'].iloc[0]
    categories = book_data['categories'].iloc[0]
    avg_sentiment = book_data['sentiment_score'].mean()

    print(f"Book Title: {title_name}")
    print(f"Authors: {authors}")
    print(f"Categories: {categories}")
    print(f"Average Sentiment Score: {avg_sentiment}")


# Filder out dataframe by catagory so only books of said catagory are printed
def filter_books_by_category(category, books_user_ratings_df, book_id):
    filtered_books = books_user_ratings_df[
        books_user_ratings_df['categories'].str.contains(category, case=False, na=False)
    ]
    book_row = books_user_ratings_df[books_user_ratings_df['Title'] == book_id]
    filtered_books = pd.concat([filtered_books, book_row]).drop_duplicates().reset_index(drop=True)
    return filtered_books if not filtered_books.empty else None


# Set category and title
example_category = "Country life"
user_category = example_category

exampleTitle = "Mini-mysteries"
user_title = exampleTitle

# Filter category
filtered_books_df = filter_books_by_category(user_category, books_user_ratings_df, user_title)

book_id = get_book_id_from_title(user_title, filtered_books_df)

# Ensure the book exists
if book_id is not None:

    # Run functions
    similar_ids = find_similar_books(book_id, X, filtered_books_df, k=5)

    book_title = book_titles[book_id]

    print(f"Since you read {book_title}:")
    # Loop through 10 closest books and print out their
    for i in similar_ids:
        get_book_details(book_titles[i])