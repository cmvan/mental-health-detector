import argparse
import random

import torch
from torch.utils.data import Dataset


class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = "\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = "\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

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

        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        padding = self.PAD_CHAR * (self.block_size + 1 - len(masked_string))
        masked_string = masked_string + padding

        input = masked_string[:-1]
        output = masked_string[1:]
        x_str = [self.stoi[s] for s in input]
        y_str = [self.stoi[s] for s in output]
        x = torch.tensor(x_str)
        y = torch.tensor(y_str)
        return x, y


class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = "\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = "\u25A1" # the empty square character, for pad
        self.itos = pretraining_dataset.itos
        self.stoi = pretraining_dataset.stoi
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]

        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y


# Code below is strictly for your debugging purposes; feel free to modify
# as desired.

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt', encoding='utf-8').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('birth_places_train.tsv', encoding='utf-8').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt', encoding='utf-8').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError(f"Unknown dataset type in command line args: {args.dataset_type}")
