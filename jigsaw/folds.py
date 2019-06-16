import argparse
import json
from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd

from .metrics import IDENTITY_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv('data/train.csv')
    df = pd.concat([
        df[df['identity_annotator_count'] > 0],
        df[df['identity_annotator_count'] == 0].sample(frac=0.29),
    ]).sample(frac=1)
    print('Identity labeled', (df['identity_annotator_count'] > 0).mean())
    ys = df[IDENTITY_COLUMNS + ['target']].fillna(0).values
    ys = ys > 0.5
    kfold = MultilabelStratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=42)
    folds = [list(map(int, df.iloc[valid_ids]['id'].values))
             for _, valid_ids in kfold.split(df, ys)]
    print('fold sizes', list(map(len, folds)))
    Path('data/folds.json').write_text(json.dumps(folds, indent=4))


if __name__ == '__main__':
    main()
