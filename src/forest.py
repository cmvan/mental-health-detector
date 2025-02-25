import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import matplotlib.pyplot as plt
from plot_confusion import plot_confusion_matrix


# loading datasets
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load the datasets from the parent directory
train_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_train.tsv"), sep="\t", header=None, names=["text", "class"], nrows=34811)
dev_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_dev.tsv"), sep="\t", header=None, names=["text", "class"], nrows=34811)
test_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_test.tsv"), sep="\t", header=None, names=["text", "class"], nrows=34811)


X_train = train_df['text']
y_train = train_df['class']
X_test = test_df['text']
y_test = test_df['class']


vectorizer = TfidfVectorizer(max_features=3500)  
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()


rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth = 20)
rf_classifier.fit(X_train_tfidf, y_train)
y_pred = rf_classifier.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"accuracy: {accuracy:.2f}")
print("\classification report:\n", classification_rep)


plot_confusion_matrix(y_test, y_pred)

