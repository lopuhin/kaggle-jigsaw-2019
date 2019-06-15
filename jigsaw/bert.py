"""
Initial version based on
https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila/
"""
import argparse
from collections import defaultdict
import json
from functools import partial
import shutil
from pathlib import Path

from apex import amp
import json_log_plots
import numpy as np
import pandas as pd
from pytorch_pretrained_bert import (
    BertTokenizer, BertForSequenceClassification, BertAdam,
)
import torch
from torch import nn
from torch import multiprocessing
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
import tqdm

from .metrics import compute_bias_metrics_for_model
from .utils import DATA_ROOT, ON_KAGGLE


device = torch.device('cuda')


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('run_root')
    arg('--train-size', type=int)
    arg('--valid-size', type=int)
    arg('--bert', default='bert-base-uncased')
    arg('--train-seq-length', type=int, default=224)
    arg('--test-seq-length', type=int, default=296)
    arg('--epochs', type=int, default=2)
    arg('--validation', action='store_true')
    arg('--submission', action='store_true')
    arg('--checkpoint', type=int)
    arg('--clean', action='store_true')
    arg('--fold', type=int, default=0)
    arg('--bucket', type=int, default=1)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if args.clean and run_root.exists():
        if input(f'Clean "{run_root.absolute()}"? ') == 'y':
            shutil.rmtree(run_root)
    do_train = not (args.submission or args.validation)
    if do_train:
        if run_root.exists():
            parser.error(f'{run_root} exists')
        run_root.mkdir(exist_ok=True, parents=True)
        params_str = json.dumps(vars(args), indent=4)
        print(params_str)
        (run_root / 'params.json').write_text(params_str)
        shutil.copy(__file__, run_root)

    print('Loading tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    y_columns = ['target']
    print('Loading model...')
    model = BertForSequenceClassification.from_pretrained(
        args.bert, num_labels=len(y_columns))
    model = model.to(device)

    model_path = run_root / 'model.pt'
    optimizer_path = run_root / 'optimizer.pt'
    best_model_path = run_root / 'model-best.pt'

    if args.submission:
        model.load_state_dict(torch.load(best_model_path))
        make_submission(model=model, tokenizer=tokenizer,
                        run_root=run_root, max_seq_length=args.test_seq_length)
        return

    train_pkl_path = DATA_ROOT / 'train.pkl'
    if not train_pkl_path.exists():
        pd.read_csv(DATA_ROOT / 'train.csv').to_pickle(train_pkl_path)
    df = pd.read_pickle(train_pkl_path)
    df = preprocess_df(df)

    folds = json.loads((DATA_ROOT / 'folds.json').read_text())
    valid_index = df['id'].isin(folds[args.fold])
    df_train, df_valid = df[~valid_index], df[valid_index]
    if args.train_size and len(df_train) > args.train_size:
        df_train = df_train.sample(n=args.train_size, random_state=42)
    if args.valid_size and len(df_valid) > args.valid_size:
        df_valid = df_valid.sample(n=args.valid_size, random_state=42)

    x_valid = tokenize_lines(
        df_valid.pop('comment_text'), args.test_seq_length, tokenizer)
    y_valid = df_valid[y_columns].values
    print(f'X_valid.shape={x_valid.shape} y_valid.shape={y_valid.shape}')

    criterion = nn.BCEWithLogitsLoss()

    def _run_validation():
        return validation(
            model=model, criterion=criterion,
            x_valid=x_valid, y_valid=y_valid, df_valid=df_valid)

    def _save(step, model, optimizer):
        torch.save(model.state_dict(), model_path)
        torch.save({'optimizer': optimizer.state_dict(), 'step': step},
                   optimizer_path)

    if args.validation:
        model.load_state_dict(torch.load(best_model_path))
        metrics = _run_validation()
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f'{v:.4f}  {k}')
    else:
        x_train = tokenize_lines(
            df_train.pop('comment_text'), args.train_seq_length, tokenizer)
        y_train = df_train[y_columns].values
        print(f'X_train.shape={x_train.shape} y_train.shape={y_train.shape}')

        best_auc = 0
        step = optimizer = None
        try:
            for model, optimizer, epoch_pbar, loss, step in train(
                    model=model, criterion=criterion,
                    x_train=x_train, y_train=y_train, epochs=args.epochs,
                    yield_steps=args.checkpoint or len(y_valid) // 8,
                    bucket=args.bucket,
                    ):
                if step == 0:
                    continue  # step 0 allows saving on Ctrl+C from the start
                _save(step, model, optimizer)
                metrics = _run_validation()
                metrics['loss'] = loss
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    shutil.copy(model_path, best_model_path)
                epoch_pbar.set_postfix(valid_loss=f'{metrics["valid_loss"]:.4f}',
                                       auc=f'{metrics["auc"]:.4f}')
                json_log_plots.write_event(run_root, step=step, **metrics)
        except KeyboardInterrupt:
            if step is not None and optimizer is not None:
                print('Ctrl+C pressed, saving checkpoint')
                _save(step, model, optimizer)
            raise


def validation(*, model, criterion, x_valid, y_valid, df_valid):
    valid_dataset = TensorDataset(
        torch.tensor(x_valid, dtype=torch.long),
        torch.tensor(y_valid, dtype=torch.float),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    valid_preds, losses = [], []
    model.eval()
    for i, (x_batch, y_batch) in enumerate(
            tqdm.tqdm(valid_loader, desc='validation', leave=False)):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_pred = model(x_batch, attention_mask=x_batch > 0, labels=None)
            loss = criterion(y_pred, y_batch)
        losses.append(float(loss.item()))
        valid_preds.extend(y_pred[:, 0].cpu().squeeze().numpy())
    model.train()

    df_valid = df_valid.copy()
    df_valid['y_pred'] = torch.sigmoid(torch.tensor(valid_preds)).numpy()

    metrics = compute_bias_metrics_for_model(df_valid, 'y_pred')
    metrics['valid_loss'] = np.mean(losses)
    return metrics


def train(
        *, model, criterion, x_train, y_train, epochs, yield_steps, bucket,
        lr=2e-5,
        batch_size=32,
        accumulation_steps=2,
        ):
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float))

    model.zero_grad()
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]

    num_train_optimization_steps = int(
        epochs * len(train_dataset) / (batch_size * accumulation_steps))
    print(f'Starting training for {num_train_optimization_steps:,} steps, '
          f'checkpoint interval {yield_steps:,}')
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=lr,
        warmup=0.05,
        t_total=num_train_optimization_steps)

    model, optimizer = amp.initialize(
        model, optimizer, opt_level='O1', verbosity=0)
    model.train()

    if bucket:
        sampler = RandomSampler(train_dataset)
        batch_sampler = BucketBatchSampler(sampler, batch_size, drop_last=False)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

    smoothed_loss = None
    step = 0
    epoch_pbar = tqdm.trange(epochs)

    def _state():
        return model, optimizer, epoch_pbar, smoothed_loss, step * batch_size

    yield _state()

    for _ in epoch_pbar:
        optimizer.zero_grad()
        pbar = tqdm.tqdm(train_loader, leave=False)
        for x_batch, y_batch in pbar:
            step += 1
            if bucket:
                x_batch, y_batch = trim_tensors([x_batch, y_batch])
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch, attention_mask=x_batch > 0, labels=None)
            loss = criterion(y_pred, y_batch)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if smoothed_loss is not None:
                smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss.item()
            else:
                smoothed_loss = loss.item()
            pbar.set_postfix(loss=f'{smoothed_loss:.4f}')

            if step % yield_steps == 0:
                yield _state()


def tokenize_lines(texts, max_seq_length, tokenizer):
    all_tokens = []
    worker = partial(
        tokenize, max_seq_length=max_seq_length, tokenizer=tokenizer)
    with multiprocessing.Pool(processes=4 if ON_KAGGLE else 16) as pool:
        for tokens in tqdm.tqdm(pool.imap(worker, texts, chunksize=100),
                                total=len(texts), desc='tokenizing'):
            all_tokens.append(tokens)
    n_max_len = sum(t[-1] != 0 for t in all_tokens)
    print(f'{n_max_len / len(texts):.1%} texts are '
          f'at least {max_seq_length} tokens long')
    return np.array(all_tokens)


def tokenize(text, max_seq_length, tokenizer):
    max_seq_length -= 2  # cls and sep
    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[:max_seq_length]
    return (tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_a + ['[SEP]']) +
            [0] * (max_seq_length - len(tokens_a)))


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'target' in df.columns:
        df = df.fillna(0)  # FIXME hmmm is this ok?
        df['target'] = (df['target'] >= 0.5).astype(float)
    df['comment_text'] = df['comment_text'].astype(str).fillna('DUMMY_VALUE')
    return df


def make_submission(*, model, tokenizer, run_root: Path, max_seq_length: int):
    df = pd.read_csv(DATA_ROOT / 'test.csv')
    df = preprocess_df(df)
    x_test = tokenize_lines(df.pop('comment_text'), max_seq_length, tokenizer)
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_preds = []
    model.eval()
    for i, (x_batch, ) in enumerate(
            tqdm.tqdm(test_loader, desc='submission', leave=False)):
        x_batch = x_batch.to(device)
        with torch.no_grad():
            y_pred = model(x_batch, attention_mask=x_batch > 0, labels=None)
        test_preds.extend(y_pred[:, 0].cpu().squeeze().numpy())
    model.train()

    df['prediction'] = torch.sigmoid(torch.tensor(test_preds)).numpy()
    root = Path('.') if ON_KAGGLE else run_root
    path = root / 'submission.csv'
    df.to_csv(path, index=None)
    print(f'Saved submission to {path}')


class BucketBatchSampler(BatchSampler):
    def __iter__(self):
        k = 8
        buckets = defaultdict(list)
        lengths = (self.sampler.data_source.tensors[0] == 0).sum(dim=1).numpy()
        for idx in self.sampler:
            buckets[binned_length(lengths[idx], k)].append(idx)

        rng = np.random.RandomState()
        for i in range(len(self)):
            batch = []
            while len(batch) < self.batch_size:
                if not batch:
                    candidates = list(buckets)
                p = np.array([len(buckets[x]) for x in candidates])
                if p.sum() == 0:
                    if len(candidates) == len(buckets):
                        assert i == len(self) - 1
                        break
                    else:
                        candidates = list(buckets)
                        continue
                length = rng.choice(candidates, p=p / p.sum())
                idx = buckets[length].pop()
                if not batch:
                    # control efficiency vs. randomness
                    if rng.rand() > 0.2:
                        candidates = [x for x in buckets
                                      if abs(x - length) <= 2 * k]
                batch.append(idx)

            yield batch


def binned_length(length: int, k=8) -> int:
    length = int(length)
    binned = max(k, k * (length // k + (length % k > 0)))
    assert binned % k == 0 and binned >= length and binned > 0
    return binned


def trim_tensors(tsrs):
    max_len = binned_length(int(torch.max(torch.sum(tsrs[0] != 0, 1))))
    tsrs = [tsr[:, :max_len] for tsr in tsrs]
    return tsrs


if __name__ == '__main__':
    main()
