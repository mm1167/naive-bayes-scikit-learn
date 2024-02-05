# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report

data = np.loadtxt('dat.csv', delimiter=',')

# Generate a larger dataset (replace with your own dataset)
# In this example, we generate random data for demonstration purposes.
# You should replace this with your actual dataset.
X = data[:, :-1]  # Features (all columns except the last one)
y = data[:, -1]   # Labels (last column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Create a Gaussian Naive Bayes classifier
cnb = ComplementNB()

# Train the classifier on the training data
cnb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = cnb.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for more detailed evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))