"""
SVM implementation
"""

import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


# loading datasets
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load the datasets from the parent directory
train_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_train.tsv"), sep="\t", header=None, names=["text", "class"])
dev_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_dev.tsv"), sep="\t", header=None, names=["text", "class"])
test_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_test.tsv"), sep="\t", header=None, names=["text", "class"])

# creating SVC model now
model = SVC(kernel='rbf')


# to extract features
vectorizer = TfidfVectorizer(max_features=5000)  
X_train = vectorizer.fit_transform(train_df["text"]).toarray()
y_train = train_df["class"]
X_test = vectorizer.transform(test_df["text"]).toarray()
y_test = test_df["class"]


# train SVM model
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train, y_train)


# predict results now
y_pred = model.predict(X_test)



cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)