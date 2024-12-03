import pandas as pd
import numpy as np
from pathlib import Path

# Load your dataset (assuming it's in a CSV or similar format)
data = Path('C:\CS3820\.vs\CS3820\\complete_cleaned_data.csv')

data_df = pd.read_csv(data, index_col=0)
# Deduplicate based on titles before sampling
unique_titles = data_df.drop_duplicates(subset=["Title"])

# Randomly sample 100000 unique titles
test_data = unique_titles.sample(n=100000, random_state=42)

# Remove the sampled rows from the original dataset
train_data = data_df[~data_df["Title"].isin(test_data["Title"])]

# Save the resulting datasets
test_data.to_csv("test_dataset_unique_titles.csv", index=False)
train_data.to_csv("train_dataset_unique_titles.csv", index=False)

print(f"Training set size: {len(train_data)}, Test set size: {len(test_data)}")

