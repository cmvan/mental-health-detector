import argparse
import random

import torch
from torch.utils.data import Dataset


class CrisisDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = "\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = "\u25A1"  # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print(f'data has {data_size} characters, {vocab_size} unique.')

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        document = self.data[idx]
        min_length = 4
        max_length = int(self.block_size*7/8)
        truncate_len = random.randint(min_length, max_length)
        truncated_doc = document[:truncate_len]

        masked_length = int(truncate_len * random.betavariate(1, 3))
        mask_start = random.randint(0, truncate_len - masked_length)

        prefix = truncated_doc[:mask_start]
        masked_content = truncated_doc[mask_start:mask_start+masked_length]
        suffix = truncated_doc[mask_start+masked_length:]

        masked_string = prefix + self.MASK_CHAR + \
            suffix + self.MASK_CHAR + masked_content
        padding = self.PAD_CHAR * (self.block_size + 1 - len(masked_string))
        masked_string = masked_string + padding

        input = masked_string[:-1]
        output = masked_string[1:]
        x_str = [self.stoi[s] for s in input]
        y_str = [self.stoi[s] for s in output]
        x = torch.tensor(x_str)
        y = torch.tensor(y_str)
        return x, y


def split_data(data, valid_size=0.1, test_size=0.1):
    n = len(data)
    valid_size = int(valid_size * n)
    test_size = int(test_size * n)

    train_data = data[:-valid_size-test_size]
    valid_data = data[-valid_size-test_size:-test_size]
    test_data = data[-test_size:]

    return train_data, valid_data, test_data


if __name__ is "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=256)
    args = parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    train_data, valid_data, test_data = split_data(
        text, valid_size=0.1, test_size=0.1)
