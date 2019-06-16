Jigsaw Unintended Bias in Toxicity Classification
-------------------------------------------------

https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/

Use Python 3.6. First install appropriate PyTorch 1.1.0 package. After that::

    pip install -r requirements.txt
    cd opt/apex
    pip install -v --no-cache-dir \
        --global-option="--cpp_ext" --global-option="--cuda_ext" .
    pip install -e .

Put data into ``./data``::

    $ ls ./data
    sample_submission.csv  test.csv  train.csv

Prepare folds::

    python -m jigsaw.folds

Prepare corpus for pre-training::

    python -m jigsaw.corpus data/corpus.txt

Pre-train the model (set ``--do_lower_case`` only for uncased models!)::

    python opt/pytorch-pretrained-BERT/examples/lm_finetuning/simple_lm_finetuning.py \
         --do_train \
         --fp16 \
         --on_memory \
         --num_train_epochs 1 \
        --train_corpus data/corpus.txt \
        --bert_model bert-base-uncased \
        --do_lower_case \
        --output _runs/pretrained-bert-uncased-ep1

Commands below don't use pre-trained model yet.

Fast training for debugging::

    python -m jigsaw.bert _runs/fast-example \
        --epochs 1 --train-size 100000 --valid-size 10000

Train::

    python -m jigsaw.bert _runs/example --epochs 2

Run validation separately::

    python -m jigsaw.bert _runs/example --validation

Make submission::

    python -m jigsaw.bert _runs/example --submission

