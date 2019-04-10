import csv
from pathlib import Path
from typing import List

import sentencepiece

from .utils import ON_KAGGLE


DATA_ROOT = Path('../input/jigsaw-2019' if ON_KAGGLE else './data')

EOL = '</n>'
VOCAB_SIZE = 16000
SP_MODEL = f'sp-{VOCAB_SIZE}.model'


def prepare_sp_text():
    """ Write all non-blank lines from train into train.txt
    """
    with Path('train.txt').open('wt', encoding='utf8') as out_f:
        for text in _text_reader('train.csv'):
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    out_f.write(line)
                    out_f.write('\n')


def _text_reader(name):
    with (DATA_ROOT / name).open('rt', encoding='utf8') as f:
        for row in csv.DictReader(f):
            yield row['comment_text']


def train_sp(vocab_size=VOCAB_SIZE):
    """ Train sentencepiece model. Run prepare_sp_text() first.
    """
    sentencepiece.SentencePieceTrainer.Train(
        f'--input=train.txt '
        f'--model_prefix={SP_MODEL.split(".")[0]} '
        f'--vocab_size={vocab_size} '
        f'--character_coverage=0.99995 '
        f'--control_symbols={EOL} '
        f'--pad_id=0 '
        f'--unk_id=1 '
        f'--bos_id=2 '
        f'--eos_id=3 ')


def load_sp_model(sp_model_path=SP_MODEL):
    sp_model = sentencepiece.SentencePieceProcessor()
    assert sp_model.load(sp_model_path)
    return sp_model


def encode_comment(sp_model: sentencepiece.SentencePieceProcessor,
                   comment: str, max_len=None) -> List[int]:
    """ Encode one comment with sentencepiece model.
    """
    # TODO we can do sub-word augmentation here
    start = sp_model.PieceToId('<s>')
    end = sp_model.PieceToId('</s>')
    eol = sp_model.PieceToId(EOL)
    pad = sp_model.PieceToId('<pad>')
    encoded = [start]
    for i, line in enumerate(comment.split('\n')):
        if i:
            encoded.append(eol)
        encoded.extend(sp_model.EncodeAsIds(line))
    encoded.append(end)
    if max_len is not None:
        encoded = encoded[:max_len]
        if len(encoded) < max_len:
            encoded.extend([pad] * (max_len - len(encoded)))
    return encoded
