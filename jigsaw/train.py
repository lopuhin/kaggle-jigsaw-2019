import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.cuda
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import tqdm

from .dataset import encode_comment, load_sp_model, SP_MODEL, DATA_ROOT
from . import models


class JigsawDataset(Dataset):
    def __init__(self, df, sp_model, max_len: int):
        super().__init__()
        self.df = df
        self.sp_model = sp_model
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        comment = encode_comment(self.sp_model, item.comment_text,
                                 max_len=self.max_len)
        comment = torch.tensor(comment, dtype=torch.int64)
        target = torch.tensor([float(item.target >= 0.5)])
        return comment, target


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('run_path')
    arg('--model', default='SimpleLSTM')
    arg('--sp-model', default=SP_MODEL)
    arg('--max-len', type=int, default=250)
    arg('--batch-size', type=int, default=512)
    arg('--lr', type=float, default=1e-4)
    arg('--epochs', type=int, default=10)
    arg('--workers', type=int, default=4)
    args = parser.parse_args()

    params_string = json.dumps(vars(args), indent=4, sort_keys=True)
    run_path = Path(args.run_path)
    run_path.mkdir(exist_ok=True, parents=True)
    (run_path / 'params.json').write_text(params_string)

    sp_model = load_sp_model(args.sp_model)
    df = pd.read_pickle(DATA_ROOT / 'train.pkl')
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    train_ids, valid_ids = next(kfold.split(df))
    train_df, valid_df = df.iloc[train_ids], df.iloc[valid_ids]

    train_dataset = JigsawDataset(train_df, sp_model, args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    valid_dataset = JigsawDataset(valid_df, sp_model, args.max_len)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_cls = getattr(models, args.model)
    model = model_cls(n_vocab=len(sp_model))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in tqdm.trange(args.epochs, desc='epoch'):
        pbar = tqdm.tqdm(train_loader)
        for xs, ys in pbar:
            optimizer.zero_grad()
            xs, ys = xs.to(device), ys.to(device)
            ys_pred = model(xs)
            loss = criterion(ys_pred, ys)
            loss.backward()
            pbar.set_postfix(loss=f'{loss.item():.2f}')


if __name__ == '__main__':
    main()
