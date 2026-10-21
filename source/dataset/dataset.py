import numpy as np

from tokenizer.tokenizer import Tokenizer

class TextDataset:


    def __init__(self, token_arr, block_size:int):
        self.token_arr = token_arr
        self.block_size = block_size

        # valid input: num_of_examples = N - block_size


    def __len__(self): # number of examples = size(tokens) - blocksize
        return(len(self.token_arr) - self.block_size)

    def __getitem__(self, idx: int):
        #tokens = [10, 20, 30, 40, 50, 60]
        token_block = self.token_arr[idx:idx + self.block_size]
        shifted_block = self.token_arr[idx+1:idx + self.block_size + 1]

        return token_block, shifted_block


"""
if __name__ == "__main__":
    token_arr = np.load('data/processed/train.npy')
    t = TextDataset(token_arr, 3)
    print(t.token_arr) # (411899,) # = size, , = 1D]

    print("dataset len: ", len(t))

    x, y = t[0]

    print("original block:", x)
    print("shifted block :", y)
"""