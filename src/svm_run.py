import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Set base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load datasets
train_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_train.tsv"),
                       sep="\t", header=None, names=["text", "class"])
dev_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_dev.tsv"),
                     sep="\t", header=None, names=["text", "class"])
test_df = pd.read_csv(os.path.join(BASE_DIR, "crisis_test.tsv"),
                      sep="\t", header=None, names=["text", "class"])

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["text"]).toarray()
y_train = train_df["class"]
X_test = vectorizer.transform(test_df["text"]).toarray()
y_test = test_df["class"]

# Define an ensemble of SVMs
base_svm = SVC(kernel="rbf", probability=True, random_state=42)
ensemble_svm = BaggingClassifier(
    estimator=base_svm, n_estimators=10, n_jobs=-1, random_state=42)

# Train the ensemble model
ensemble_svm.fit(X_train, y_train)

# Predict results
y_pred = ensemble_svm.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)

# Confusion Matrix:
# [[11029   542]
#  [734 10903]]
