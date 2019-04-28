"""
https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
"""
import os
import time
import random
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

from .dataset import DATA_ROOT


def seed_everything(seed):  # FIXME remove
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


CRAWL_EMBEDDING_PATH = DATA_ROOT / 'crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = DATA_ROOT / 'glove.840B.300d.txt'
NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 220


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_model(model, train, test, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 0.6 ** epoch)
    
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        scheduler.step()
        
        model.train()
        avg_loss = 0.
        
        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)            
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        test_preds = np.zeros((len(test), output_dim))
    
        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())

            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        logging.info(
            'Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, elapsed_time))

    if enable_checkpoint_ensemble:
        test_preds = np.average(
            all_test_preds, weights=checkpoint_weights, axis=0)
    else:
        test_preds = all_test_preds[-1]
        
    return test_preds


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets, max_features):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out


def preprocess(data):
    """
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    """
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s: %(message)s')

    # Preprocessing

    logging.info('reading train and test')
    train = pd.read_csv(DATA_ROOT / 'train.csv')
    test = pd.read_csv(DATA_ROOT / 'test.csv')

    logging.info('pre-processing text')
    x_train = preprocess(train['comment_text'])
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    y_aux_train = train[['target', 'severe_toxicity', 'obscene',
                         'identity_attack', 'insult', 'threat']]
    x_test = preprocess(test['comment_text'])

    max_features = None

    logging.info('building tokenizer')
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    logging.info('applying tokenizer')
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    max_features = max_features or len(tokenizer.word_index) + 1
    logging.info(f'max_features {max_features:,}')

    logging.info('building crawl embedding matrix')
    crawl_matrix, unknown_words_crawl = build_matrix(
        tokenizer.word_index, CRAWL_EMBEDDING_PATH)
    logging.info(f'n unknown words (crawl): {len(unknown_words_crawl):,}')

    logging.info('building glove embedding matrix')
    glove_matrix, unknown_words_glove = build_matrix(
        tokenizer.word_index, GLOVE_EMBEDDING_PATH)
    logging.info(f'n unknown words (glove): {len(unknown_words_glove):,}')

    embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
    logging.info(f'embedding.matrix {embedding_matrix.shape}')

    logging.info('building tensors')
    x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
    x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()
    y_train_torch = torch.tensor(
        np.hstack([y_train[:, np.newaxis], y_aux_train]),
        dtype=torch.float32).cuda()

    # Training

    train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
    test_dataset = data.TensorDataset(x_test_torch)

    all_test_preds = []

    for model_idx in range(NUM_MODELS):
        logging.info(f'Model {model_idx}')
        seed_everything(1234 + model_idx)

        model = NeuralNet(embedding_matrix, y_aux_train.shape[-1], max_features)
        model.cuda()

        test_preds = train_model(model, train_dataset, test_dataset,
                                 output_dim=y_train_torch.shape[-1],
                                 loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
        all_test_preds.append(test_preds)

    submission = pd.DataFrame.from_dict({
        'id': test['id'],
        'prediction': np.mean(all_test_preds, axis=0)[:, 0]
    })

    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
