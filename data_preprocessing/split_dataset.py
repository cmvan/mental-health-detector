import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("suicide_detection.csv", index_col=0)

    df['text'] = df['text'].apply(lambda x: ' '.join(x.splitlines()))
    train, dev, test = np.split(df.sample(frac=1, random_state=42), [
        int(0.8*len(df)), int(0.9*len(df))])
    vocab = df['text'].unique()

    train.to_csv("crisis_train.tsv", sep="\t", header=None, index=False)
    dev.to_csv("crisis_dev.tsv", sep="\t", header=None, index=False)
    test.to_csv("crisis_test.tsv", sep="\t", header=None, index=False)

    with open("vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")
