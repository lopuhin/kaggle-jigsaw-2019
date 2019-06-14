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

Train::

    python -m jigsaw.bert _runs/example --epochs 2

Run validation separately::

    python -m jigsaw.bert _runs/example --validation

Make submission::

    python -m jigsaw.bert _runs/example --submission

