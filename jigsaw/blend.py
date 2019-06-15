import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submissions', nargs='+')
    parser.add_argument('--out', default='submission.csv')
    args = parser.parse_args()

    if len(args.submissions) < 2:
        parser.error('At least two submissions required for blend')
    dfs = []
    for path in args.submissions:
        df = pd.read_csv(path)
        dfs.append(df)

    blend_df = dfs[0].copy()
    blend_df['prediction'] = np.mean(
        [df['prediction'].values for df in dfs], axis=0)
    blend_df.to_csv(args.out)
    print(f'Saved blend to {args.out}')


if __name__ == '__main__':
    main()
