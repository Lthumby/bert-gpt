import numpy as np
from dataset.dataset import TextDataset


def get_batch(dataset: TextDataset, batch_size:int, rng: np.random.Generator):
    valid_starts = len(dataset)
    idx = rng.integers(0, len(dataset), size=batch_size)

    xlist = []
    ylist = []

    for i in idx:
        x,y = dataset[int(i)]
        xlist.append(x)
        ylist.append(y)

    xbatch = np.stack(xlist)
    ybatch = np.stack(ylist)


    return xbatch, ybatch

#tokens = np.arange(20)
#dataset = TextDataset(tokens, block_size=3)

#rng = np.random.default_rng(42)

#x, y = get_batch(dataset, batch_size=3, rng=rng)

#print("X:", x)
#print("Y:", y)
#print("X shape:", x.shape)
#print("Y shape:", y.shape)