import argparse
import json
import re

import pandas as pd
from spacy.lang.en import English
import tqdm

from .utils import DATA_ROOT


def main():
    """ Generate corpus for pre-training. See
    https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/lm_finetuning
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('output')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    df = pd.read_pickle(DATA_ROOT / 'train.pkl')
    folds = json.loads((DATA_ROOT / 'folds.json').read_text())
    df = df[~df['id'].isin(folds[args.fold])]

    n_skipped = n_texts = n_sents = 0
    with open(args.output, 'wt', encoding='utf8') as outf:
        for text in tqdm.tqdm(
                df.sample(frac=1, random_state=42)['comment_text'].values):
            try:
                doc = nlp(text)
                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    sent_text = re.sub(r'\s+', ' ', sent_text)
                    print(sent_text, file=outf)
                    n_sents += 1
                print('', file=outf)
                n_texts += 1
            except ValueError as e:
                if str(e).startswith('[E030]'):
                    n_skipped += 1
                    continue
                raise
    print(f'Stats: {n_texts:,} texts, {n_sents:,} sentences, '
          f'{n_skipped:,} skipped.')


if __name__ == '__main__':
    main()
