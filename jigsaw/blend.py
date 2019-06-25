import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd


def blend(dfs, weights: Union[str, List[float]], column: str):
    blend_df = dfs[0].copy()
    if weights:
        if isinstance(weights, str):
            weights = list(map(float, weights.split(',')))
    else:
        weights = [1] * len(dfs)
    assert len(weights) == len(dfs)
    weights = [w / sum(weights) for w in weights]
    blend_df[column] = np.mean(
        [w * df[column].values for w, df in zip(weights, dfs)],
        axis=0)
    return blend_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submissions', nargs='+')
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--weights', help='comma separated')
    args = parser.parse_args()

    if len(args.submissions) < 2:
        parser.error('At least two submissions required for blend')
    if args.weights:
        all_weights = list(map(float, args.weights.split(',')))
    else:
        all_weights = [1] * len(args.submissions)
    dfs = []
    weights = []
    for path, w in zip(args.submissions, all_weights):
        if Path(path).exists():
            df = pd.read_csv(path)
            dfs.append(df)
            weights.append(w)
        else:
            print(f'missing file {path}')

    blend_df = blend(dfs, weights, 'prediction')
    blend_df.to_csv(args.out, index=None)
    print(f'Saved blend to {args.out}')


if __name__ == '__main__':
    main()
