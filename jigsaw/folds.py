import json
from pathlib import Path

import pandas as pd

from .metrics import IDENTITY_COLUMNS


def main():
    df = pd.read_pickle('data/train.pkl')
    annot_df = (df[df['identity_annotator_count'] > 0]
                .sample(n=48660, random_state=13))
    not_annot_df = (df[df['identity_annotator_count'] == 0]
                    .sample(n=48660, random_state=13))
    valid_df = pd.concat([annot_df, not_annot_df])
    train_df = df[~df['id'].isin(valid_df['id'])]
    assert len(valid_df) + len(train_df) == len(df)
    train_df_check = pd.concat([
        train_df[train_df['identity_annotator_count'].fillna(0) > 0],
        train_df[train_df['identity_annotator_count'].fillna(0) == 0]
        .sample(frac=0.27)
    ])
    for col in ['target', 'identity_annotator_count'] + IDENTITY_COLUMNS:
        print(f'{col:<40} '
              f'{(valid_df[col].fillna(0) >= 0.5).values.mean():.4f} '
              f'{(train_df_check[col].fillna(0) >= 0.5).values.mean():.4f} ')
    folds = [list(map(int, valid_df['id'].values))]
    print('fold sizes', list(map(len, folds)))
    Path('data/folds.json').write_text(json.dumps(folds, indent=4))


if __name__ == '__main__':
    main()
