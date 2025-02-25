import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from plot_confusion import plot_confusion_matrix


# Define file paths (move one directory up)
base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
parent_path = os.path.abspath(os.path.join(base_path, ".."))  # Move one level up

crisis_terms_file = os.path.join(parent_path, "crisis_terms.txt")
input_csv_file = os.path.join(parent_path, "Suicide_Detection.csv")
output_csv_file = os.path.join(parent_path, "classified_test.csv") 


def load_crisis_terms(filename):
    with open(filename, "r") as file:
        return [line.strip().lower() for line in file if line.strip()]


# Load crisis terms
crisis_terms = load_crisis_terms(crisis_terms_file)


# Function to classify text based on crisis terms (handles multi-word terms)
def classify_text(text):
    if pd.isna(text):  # Handle missing values
        return "N/A"
    text_lower = text.lower()
    if any(term in text_lower for term in crisis_terms):  # Check for phrase presence
        return "suicide"
    return "non-suicide"


chunk_size = 10000  # Adjust based on memory constraints
chunks = []
for chunk in pd.read_csv(input_csv_file, chunksize=chunk_size):
    if "text" in chunk.columns:  
        chunk["classification"] = chunk["text"].apply(classify_text)
        chunks.append(chunk)


df_result = pd.concat(chunks, ignore_index=True)
df_result.to_csv(output_csv_file, index=False)


print("Classification completed. First few rows of output:")
print(df_result.head())


TP = ((df_result["classification"] == "suicide") & (df_result["class"] == "suicide")).sum()
FP = ((df_result["classification"] == "suicide") & (df_result["class"] == "non-suicide")).sum()
TN = ((df_result["classification"] == "non-suicide") & (df_result["class"] == "non-suicide")).sum()
FN = ((df_result["classification"] == "non-suicide") & (df_result["class"] == "suicide")).sum()

# Calculate Sensitivity (Recall), Specificity, PPV, and NPV
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
npv = TN / (TN + FN) if (TN + FN) > 0 else 0

# Print the metrics
print("\nPerformance Metrics**")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Positive Predictive Value (PPV): {ppv:.4f}")
print(f"Negative Predictive Value (NPV): {npv:.4f}")

accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy:.4f}")


y_true = df_result['class']
y_pred = df_result['classification']

plot_confusion_matrix(y_true, y_pred)

