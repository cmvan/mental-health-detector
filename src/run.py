import random
import argparse

import dataset
import models
import trainer
import utils

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

random.seed(0)

argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
argp.add_argument('variant', help="Choose vanilla or rope")
argp.add_argument('pretrain_corpus_path', default=None)
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--writing_params_path',default=None)
argp.add_argument('--finetune_corpus_path', default=None)
argp.add_argument('--eval_corpus_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--pretrain_lr', default=6e-3, type=float)
argp.add_argument('--finetune_lr', default=6e-4, type=float)
argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                  default='run')
args = argp.parse_args()

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
elif torch.backends.mps.is_available() and args.variant == 'vanilla':
    device = 'mps'

# TensorBoard training log
writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
    args.function,
    args.tb_expt_name,
    args.variant,
    args.pretrain_lr,
    args.finetune_lr))


block_size = 128
text = open(args.pretrain_corpus_path, encoding='utf-8').read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)


mconf = models.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256)

# define models.
# note: models should moved to device defined on lines 30-34.

model = None
if args.variant == 'vanilla':
    model = models.GPT(mconf)
    model.to(device)
elif args.variant == 'rope':
    mconf.rope = True
    model = models.GPT(mconf)
    model.to(device)
else:
    raise ValueError("Unknown model variant")

print('Model on device: ', next(model.parameters()).device)

# Perform pretraining, finetuning, or evaluation
if args.function == 'pretrain':
    assert args.writing_params_path is not None
    ptrainconfig = trainer.TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=args.pretrain_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=650*len(pretrain_dataset)*block_size,
        num_workers=4,
        writer=writer,
    )
    pretrain = trainer.Trainer(model, pretrain_dataset, test_dataset=None, config=ptrainconfig)
    pretrain.train()
    torch.save(model.state_dict(), args.writing_params_path)
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    text = open(args.finetune_corpus_path, encoding="utf-8").read()
    fine_data = dataset.NameDataset(pretrain_dataset, text)
    trainconfig = None
    if args.reading_params_path is not None:
        model.load_state_dict(torch.load(args.reading_params_path))
        trainconfig = trainer.TrainerConfig(
            max_epochs=10,
            batch_size=256,
            learning_rate=args.finetune_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=4,
            writer=writer,
        )
    else:
        trainconfig = trainer.TrainerConfig(
            max_epochs=75,
            batch_size=256,
            learning_rate=args.finetune_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=4,
            writer=writer,
        )
    finetrain = trainer.Trainer(model, fine_data, test_dataset=None, config=trainconfig)
    finetrain.train()
    torch.save(model.state_dict(), args.writing_params_path)
elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x],
                             dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print(f'Correct: {correct} out of {total}: {correct/total*100}%')
    else:
        print(f'Predictions written to {args.outputs_path}; no targets provided')
