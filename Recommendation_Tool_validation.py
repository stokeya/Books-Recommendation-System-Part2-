import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = Path('C:\CS3820\.vs\CS3820\\complete_cleaned_data.csv')

data_df = pd.read_csv(data, index_col=0)

# Ensure the dataset has no missing values
data_df = data_df.dropna()

# Split into features (X) and target (y)
# Example features: 'User_id', 'Title', 'review/helpfulness'
# Example target: 'review/score' (actual rating)
X = data_df[['User_id', 'Title', 'review/helpfulness']]  
y = data_df['review/score']

# Initialize LabelEncoder
label_encoder_user = LabelEncoder()
label_encoder_title = LabelEncoder()
label_encoder_helpfulness = LabelEncoder()
# Encode 'User_id' and 'Title' using LabelEncoder
X = X.copy()  # Explicitly make a copy of X to avoid the warning
X.loc[:, 'User_id'] = label_encoder_user.fit_transform(X['User_id'])
X.loc[:, 'Title'] = label_encoder_title.fit_transform(X['Title'])
X.loc[:, 'review/helpfulness'] = label_encoder_helpfulness.fit_transform(X['review/helpfulness'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize K-Nearest Neighbors (KNN) model
knn = KNeighborsClassifier(n_neighbors=5)  # Use 5 neighbors (can tune this value)

# Train the KNN model
knn.fit(X_train, y_train)

# Predict ratings for the test set
y_pred = knn.predict(X_test)

# Compute Regression Metrics (1-5 scale)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Regression Metrics (1-5 Scale):\nMSE: {mse}\nMAE: {mae}")

# Compute Classification Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Classification Metrics:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}")

# Visualize actual vs. predicted ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.title("Actual Ratings vs. Predicted Ratings (KNN)")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.grid()
plt.show()
