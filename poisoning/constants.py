from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

data_dir = _PROJECT_ROOT / 'data'
model_dir = _PROJECT_ROOT / 'models'
batch_size = 64
epsilon = 1e-7
