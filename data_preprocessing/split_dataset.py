import pandas as pd
import numpy as np


def split_test_set(test_df):
    suicide_df = test_df[test_df["class"] == "suicide"]
    non_suicide_df = test_df[test_df["class"] == "non-suicide"]

    # Compute new target sizes for the 32%-68% split
    total_test_samples = len(test_df)
    desired_suicide_count = int(0.32 * total_test_samples)
    desired_non_suicide_count = int(0.68 * total_test_samples)

    # Sample each class to match the desired ratio
    suicide_sample = suicide_df.sample(n=min(desired_suicide_count, len(suicide_df)), random_state=42)
    non_suicide_sample = non_suicide_df.sample(n=min(desired_non_suicide_count, len(non_suicide_df)), random_state=42)

    # Combine and shuffle
    balanced_test_set = pd.concat([suicide_sample, non_suicide_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_test_set


    # print(f"New test set size: {len(balanced_test_set)}")
    # print(f"Suicide samples: {len(suicide_sample)}, Non-suicide samples: {len(non_suicide_sample)}")



if __name__ == "__main__":
    df = pd.read_csv("../Suicide_Detection.csv", index_col=0)

    df['text'] = df['text'].apply(lambda x: ' '.join(x.splitlines()))
    train, dev, test = np.split(df.sample(frac=1, random_state=42), [
        int(0.8*len(df)), int(0.9*len(df))])
    vocab = df['text'].unique()

    train.to_csv("crisis_train.tsv", sep="\t", header=None, index=False)
    dev.to_csv("crisis_dev.tsv", sep="\t", header=None, index=False)
    test.to_csv("crisis_test.tsv", sep="\t", header=None, index=False)

    # Save the new test set
    balanced_test_set = split_test_set(test)
    balanced_test_set.to_csv("balanced_test_set.csv", index=False)

    with open("vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")
