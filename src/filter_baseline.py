import pandas as pd
import os


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
    if "text" in chunk.columns:  # Replace with the actual column name
        chunk["classification"] = chunk["text"].apply(classify_text)
        chunks.append(chunk)


df_result = pd.concat(chunks, ignore_index=True)
df_result.to_csv(output_csv_file, index=False)


print("Classification completed. First few rows of output:")
print(df_result.head())

if "suicide" or "non-suicide" in df_result.columns:  # Replace with the actual name of the classification column
    # Compare the new "Label" column with the pre-existing "X or not X" column
    correct_matches = (df_result["class"] == df_result["classification"]).sum()
    incorrect_matches = (df_result["class"] != df_result["classification"]).sum()

    # Calculate False Positives and False Negatives
    false_positives = ((df_result["classification"] == "suicide") & (df_result["class"] == "non-suicide")).sum()
    false_negatives = ((df_result["classification"] == "non-suicide") & (df_result["class"] == "suicide")).sum()

    total_rows = len(df_result)
    false_positive_percentage = (false_positives / total_rows) * 100
    false_negative_percentage = (false_negatives / total_rows) * 100
    #accuracy = (correct_matches / total_rows) * 100
    
    # Print the comparison report
    print("\n📝 **Comparison Report**")
    print(f"✅ Number of correct matches: {correct_matches}")
    print(f"❌ Number of incorrect matches: {incorrect_matches}")
    print(f"✅ Accuracy (correct matches / total): {correct_matches / len(df_result):.4f}")
    print(f"❌ False Positives (FP): {false_positive_percentage}")
    print(f"❌ False Negatives (FN): {false_negative_percentage}")
