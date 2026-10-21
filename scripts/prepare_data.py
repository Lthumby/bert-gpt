import numpy as np
from pathlib import Path

from tokenizer.tokenizer import Tokenizer 

# Eventually change to a config
PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "source"
DATA_PATH = PROJECT_DIR / "data" / "raw" / "input.txt"
MERGES_PATH = SOURCE_DIR /  "tokenizer" / "merges.json"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


t = Tokenizer()
t.load(MERGES_PATH)

text = DATA_PATH.read_text(encoding="utf-8")

split_idx = int(len(text) * 0.9)

train_text = text[:split_idx]
val_text = text[split_idx:]

encoded_train = t.encode(train_text)
encoded_val = t.encode(val_text)

np_train = np.array(encoded_train, dtype=np.uint16)
np_val = np.array(encoded_val, dtype=np.uint16)

np.save(OUTPUT_DIR / "train.npy", np_train)
np.save(OUTPUT_DIR / "val.npy", np_val)

print("Training shape:", np_train.shape)
print("Validation shape:", np_val.shape)
print("Training dtype:", np_train.dtype)
print("First 10 tokens:", np_train[:10])

