import pandas as pd
import argparse
import utils


def load_crisis_terms(crisis_terms_file):
    """ Load crisis terms into a set for fast lookup """
    return set(pd.read_csv(crisis_terms_file, header=None)[0].str.lower())


def contains_crisis_term(text, crisis_terms):
    """ Check if any crisis term appears in the text """
    if pd.isna(text):
        return False
    text_lower = text.lower()
    return any(term in text_lower for term in crisis_terms)


def filter_crisis_terms(classify_text, input, output):
    chunk_size = 10000
    chunks = []
    for chunk in pd.read_csv(input, chunksize=chunk_size):
        if "text" in chunk.columns:
            chunk["classification"] = chunk["text"].apply(classify_text)
            chunks.append(chunk)
    res_df = pd.concat(chunks, ignore_index=True)
    res_df.to_csv(output, index=False)
    return res_df


def filter_crisis_terms(classify_text, input, output):
    chunk_size = 10000
    chunks = []
    for chunk in pd.read_csv(input, chunksize=chunk_size):
        if "text" in chunk.columns:
            chunk["classification"] = chunk["text"].apply(classify_text)
            chunks.append(chunk)
    res_df = pd.concat(chunks, ignore_index=True)
    res_df.to_csv(output, index=False)
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crisis_terms", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    crisis_terms_file = args.crisis_terms
    input = args.input
    output = args.output

    res_df = filter_crisis_terms(classify_text, input, output)

    print("Classification completed. First few rows of output:")
    print(res_df.head())

    y_true, y_pred = res_df['class'], res_df['classification']
    utils.evaluate_model(y_true, y_pred)
