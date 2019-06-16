import argparse
import json
from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd

from .metrics import IDENTITY_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv('data/train.csv')
    ys = df[IDENTITY_COLUMNS + ['target']].fillna(0).values
    ys = ys > 0.5
    kfold = MultilabelStratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=42)
    folds = [list(map(int, df.iloc[valid_ids]['id'].values))
             for _, valid_ids in kfold.split(df, ys)]
    Path('data/folds.json').write_text(json.dumps(folds, indent=4))


if __name__ == '__main__':
    main()
