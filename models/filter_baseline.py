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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crisis_terms", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Load crisis terms
    crisis_terms = load_crisis_terms(args.crisis_terms)

    chunk_size = 10000  # Adjust based on memory constraints
    filtered_chunks = []

    # Process CSV in chunks, filtering out non-matching rows
    for chunk in pd.read_csv(args.input, sep="\t", names=["text", "class"], chunksize=chunk_size):
        if "text" in chunk.columns:
            # Keep only rows where the text contains a crisis term
            filtered_chunk = chunk[chunk["text"].apply(lambda x: contains_crisis_term(x, crisis_terms))]

            # Append only non-empty chunks
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)

    # Combine all filtered chunks into a single dataframe
    if filtered_chunks:
        filtered_res_df = pd.concat(filtered_chunks, ignore_index=True)
        filtered_res_df.to_csv(args.output, index=False)
        print(f"Filtered crisis-only data saved to {args.output}")
    else:
        print("No matching rows found. Empty output file generated.")

    print("Processing complete.")
