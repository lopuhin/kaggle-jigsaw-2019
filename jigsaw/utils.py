from pathlib import Path
import os


ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ
DATA_ROOT = Path('../input/jigsaw-2019' if ON_KAGGLE else './data')
