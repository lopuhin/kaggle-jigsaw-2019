import argparse
import json
from pathlib import Path
import statistics
import shutil

import fastText
import json_log_plots
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.cuda
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import tqdm

from ..utils import DATA_ROOT
from .dataset import encode_comment, load_sp_model, SP_MODEL
from ..metrics import compute_bias_metrics_for_model, MAIN_METRICS
from . import models


class JigsawDataset(Dataset):
    AUX_TARGETS = [
        'target',
        'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

    def __init__(self, df, sp_model, max_len: int):
        super().__init__()
        self.df = df
        self.sp_model = sp_model
        self.max_len = max_len
        self.has_target = 'target' in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        comment = encode_comment(self.sp_model, item.comment_text,
                                 max_len=self.max_len)
        comment = torch.tensor(comment, dtype=torch.int64)
        if self.has_target:
            target = torch.tensor(
                [float(item.target >= 0.5)] +
                [getattr(item, col) for col in self.AUX_TARGETS])
            assert not torch.isnan(target).any()
            return comment, target
        else:
            return comment


def collate_fn(inputs):
    has_target = not isinstance(inputs[0], torch.Tensor)
    if has_target:
        inputs = [(i, x, y) for i, (x, y) in enumerate(inputs)]
    else:
        inputs = list(enumerate(inputs))
    inputs = sorted(inputs, key=lambda x: len(x[1]), reverse=True)
    inverse = {x[0]: i for i, x in enumerate(inputs)}
    indices = [inverse[i] for i in range(len(inverse))]
    comments = [x[1] for x in inputs]
    if has_target:
        targets = torch.stack([x[2] for x in inputs])
    else:
        targets = None
    lengths = torch.tensor(list(map(len, comments)))
    comments = pad_sequence(comments, batch_first=True)
    collated = (comments, lengths, indices)
    if has_target:
        collated += (targets,)
    return collated


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('action', choices=['train', 'validate', 'submit'])
    arg('run_path')
    arg('--model', default='SimpleLSTM')
    arg('--sp-model', default=SP_MODEL)
    arg('--max-len', type=int, default=250)
    arg('--batch-size', type=int, default=512)
    arg('--lr', type=float, default=3e-4)
    arg('--lr-power-base', type=float, default=1)
    arg('--epochs', type=int, default=10)
    arg('--workers', type=int, default=4)
    arg('--validate-every', type=int, default=1000)
    arg('--clean', action='store_true')
    arg('--n-embed', type=int, default=128)
    arg('--embed-init')
    arg('--embed-freeze', type=int, default=0)
    args = parser.parse_args()

    run_path = Path(args.run_path)
    params_path = run_path / 'params.json'
    save_path = run_path / 'net.pt'
    action = args.action
    if action == 'train':
        params = vars(args)
        params_string = json.dumps(params, indent=4, sort_keys=True)
        print(params_string)
        if args.clean and run_path.exists():
            shutil.rmtree(run_path)
        run_path.mkdir(exist_ok=True, parents=True)
        params_path.write_text(params_string)
        for p in Path('jigsaw').glob('*.py'):
            shutil.copy(p, run_path)
    else:
        # args are ignored
        params = json.loads(params_path.read_text())
    del args

    sp_model = load_sp_model(params['sp_model'])
    if action != 'submit':
        train_pkl_path = DATA_ROOT / 'train.pkl'
        if not train_pkl_path.exists():
            pd.read_csv(DATA_ROOT / 'train.csv').to_pickle(train_pkl_path)
        df = pd.read_pickle(train_pkl_path)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        train_ids, valid_ids = next(kfold.split(df))
        train_df, valid_df = df.iloc[train_ids], df.iloc[valid_ids]

        train_dataset = JigsawDataset(train_df, sp_model, params['max_len'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=params['workers'],
            collate_fn=collate_fn,
        )
        valid_dataset = JigsawDataset(valid_df, sp_model, params['max_len'])
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=params['workers'],
            collate_fn=collate_fn,
        )
        print(f'train size: {len(train_dataset):,} '
              f'valid size: {len(valid_dataset):,}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_cls = getattr(models, params['model'])
    n_aux = len(JigsawDataset.AUX_TARGETS)
    model: nn.Module = model_cls(
        n_vocab=len(sp_model),
        n_embed=params['n_embed'],
        n_out=1 + n_aux,
    )
    if action == 'train':
        if params['embed_freeze']:
            model.embedding.weight.requires_grad = False
        if params['embed_init']:
            print('loading embeddings')
            ft_model = fastText.load_model(params['embed_init'])
            for i in range(len(sp_model)):
                model.embedding.weight[i] = torch.tensor(
                    ft_model.get_word_vector(
                        sp_model.IdToPiece(i).strip('▁')))
            del ft_model
        print(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    pos_weight = torch.tensor([float(n_aux)] + [1.0] * n_aux)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: params['lr_power_base'] ** epoch)
    step = 0

    def save():
        torch.save({
            'state_dict': model.state_dict(),
            'params': params,
        }, save_path)

    def train_step(xs, lengths, _,  ys):
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        ys_pred = model(xs, lengths)
        loss = criterion(ys_pred, ys)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_validation_metrics():
        losses = []
        predictions = []
        model.eval()
        with torch.no_grad():
            for xs, lengths, indices, ys in tqdm.tqdm(
                    valid_loader,
                    desc='validate', dynamic_ncols=True, leave=False):
                xs, ys = xs.to(device), ys.to(device)
                ys_pred = model(xs, lengths)
                loss = criterion(ys_pred, ys)
                losses.append(loss.item())
                predictions.extend(
                    map(float, torch.sigmoid(ys_pred[indices, 0]).data.cpu()))
        model.train()
        valid_loss_value = statistics.mean(losses)
        pred_df = valid_loader.dataset.df.copy()
        pred_df['pred'] = predictions
        metrics = compute_bias_metrics_for_model(pred_df, 'pred')
        metrics['valid_loss'] = valid_loss_value
        return metrics

    def validate():
        json_log_plots.write_event(
            run_path, step * params['batch_size'], **get_validation_metrics())

    def submit():
        test_df = pd.read_csv(DATA_ROOT / 'test.csv')
        model.eval()
        test_dataset = JigsawDataset(test_df, sp_model, params['max_len'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=params['workers'],
            collate_fn=collate_fn,
        )
        predictions = []
        with torch.no_grad():
            for xs, lengths, indices in tqdm.tqdm(
                    test_loader, dynamic_ncols=True, leave=False):
                xs = xs.to(device)
                ys = torch.sigmoid(model(xs, lengths)[indices, 0])
                predictions.extend(map(float, ys.data.cpu()))
        test_df.pop('comment_text')
        test_df['prediction'] = predictions
        test_df.to_csv('submission.csv', index=None)

    def train():
        nonlocal step
        for _ in tqdm.trange(params['epochs'], desc='epoch',
                             dynamic_ncols=True):
            lr_scheduler.step()
            pbar = tqdm.tqdm(train_loader, desc='train', dynamic_ncols=True)
            for batch in pbar:
                loss_value = train_step(*batch)
                step += 1
                pbar.set_postfix(loss=f'{loss_value:.2f}')
                json_log_plots.write_event(
                    run_path, step * params['batch_size'], loss=loss_value)
                if (params['validate_every'] and
                        step % params['validate_every'] == 0):
                    validate()
            save()
            validate()

    if action == 'train':
        try:
            train()
        except KeyboardInterrupt:
            print('Interrupted, saving...')
            save()
            exit(1)
    else:
        model.load_state_dict(
            torch.load(save_path, map_location=device)['state_dict'])
        if action == 'validate':
            valid_metrics = get_validation_metrics()
            for k in MAIN_METRICS + ['valid_loss']:
                print(f'{k:<20} {valid_metrics[k]:.4f}')
        elif action == 'submit':
            submit()


if __name__ == '__main__':
    main()
