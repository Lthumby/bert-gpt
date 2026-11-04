import numpy as np
from dataset.dataset import TextDataset


def get_batch(dataset: TextDataset, batch_size:int, rng: np.random.Generator):

    tokens = dataset.token_arr
    T = dataset.block_size

    starts = rng.integers(low=0, high=len(dataset), size=batch_size)
    offsets = np.arange(T)


    xbatch = tokens[starts[:,None] + offsets]
    ybatch = tokens[starts[:,None] + offsets + 1]


    return (xbatch.astype(np.int64), ybatch.astype(np.int64))

#tokens = np.arange(20)
#dataset = TextDataset(tokens, block_size=3)

#rng = np.random.default_rng(42)

#x, y = get_batch(dataset, batch_size=3, rng=rng)

#print("X:", x)
#print("Y:", y)
#print("X shape:", x.shape)
#print("Y shape:", y.shape)