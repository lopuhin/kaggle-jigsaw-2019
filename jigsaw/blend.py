import argparse

import numpy as np
import pandas as pd

from .metrics import blend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submissions', nargs='+')
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--weights', help='comma separated')
    args = parser.parse_args()

    if len(args.submissions) < 2:
        parser.error('At least two submissions required for blend')
    dfs = []
    for path in args.submissions:
        df = pd.read_csv(path)
        dfs.append(df)

    blend_df = blend(dfs, args.weights, 'prediction')
    blend_df.to_csv(args.out, index=None)
    print(f'Saved blend to {args.out}')


if __name__ == '__main__':
    main()
