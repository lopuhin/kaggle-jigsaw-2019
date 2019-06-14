Install
-------

Use Python 3.6. First install appropriate pytorch version. After that::

    pip install -r requirements.txt
    cd opt/apex
    pip install -v --no-cache-dir \
        --global-option="--cpp_ext" --global-option="--cuda_ext" .

