from pathlib import Path
from src.config import *
import pandas as pd
print(Path.cwd())

x_train = pd.read_csv(X_TEXT_TRAIN_PATH)
print(x_train.columns)