import json
from pathlib import Path
import regex as re
gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:


    def __init__(self):
        self.merges = {} # stores training rules
        self.vocab = {i: bytes([i]) for i in range(256)} # all mappings of bytes
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def pre_tokenization(self, text:str):
        return re.findall(gpt2_pattern, text)
    def to_byte(self, text:str):
        return list(bytes(text, 'utf-8'))

    def find_pairs(self, byte_text:list):
        freq_map = {}
        # [104, 101, 108, 108, 111]
        #  i.   i+1 -> i+1,i+2
        pair_list = []
        for i in range(len(byte_text)):
            if i < len(byte_text) - 1:
                pair_list.append((byte_text[i], byte_text[i+1]))

        return pair_list

    def freq_map(self, tokens:list[list[int]]):
        pairs = []

        freq_map = {}
        for token in tokens:
            pairs = self.find_pairs(token)
            for item in pairs:
                
                if item in freq_map:
                    freq_map[item] += 1
                else:
                    freq_map[item] = 1
        return freq_map

    def most_freq_pair(self, freq_map:dict):
        max_key, max_val = max(freq_map.items(), key=lambda item: item[1])
        return (max_key, max_val)

    def merge_op(self, tokens:list, pair:tuple, new_id:int):
        output = []
        #[1,2,3,1,2]	(1,2) → 256	[256,3,256]
        token_range = iter(range(len(tokens)))

        for i in token_range:
            
            if (i >=len(tokens) - 1):
                output.append(tokens[i])
                continue
            new_pair = (tokens[i], tokens[i+1])
            #print(f"new pair: {new_pair}")
            if new_pair == pair:
                output.append(new_id)
                next(token_range, None)
            else:
                output.append(tokens[i])
        return output


    def train(self, text:str, vocab_size:int, verbose=False):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)} # all mappings of bytes

        ID = 256

        chunks = self.pre_tokenization(text)

        token_chunks = [self.to_byte(chunk) for chunk in chunks]
        if vocab_size < ID:
            raise ValueError("# < 256")


        vocab = iter(range(vocab_size - ID))
        for i in vocab:
            
            freq_map = self.freq_map(token_chunks)
            #print(f"{i}: token list {token_chunks}")
            if not freq_map:
                break
            
            pair, count = self.most_freq_pair(freq_map)
            token_chunks = [
                self.merge_op(tokens, pair, ID) for tokens in token_chunks
            ]

            #token_list = self.merge_op(token_list, most_freq_pair[0], ID) 
            self.merges[pair] = ID
            byte_sequence = self.vocab[pair[0]] + self.vocab[pair[1]] # so engine can recognize bytes > 255 (custom bytes created)
            self.vocab[ID] = byte_sequence
            ID += 1

    def encode_chunk(self, text: str):
        token_list = self.to_byte(text)

        while True:
            pair_list = self.find_pairs(token_list)

            que = []
            for item in pair_list:
                if item in self.merges:
                    que.append(item)

            if not que:
                break

            pair = min(que, key=lambda item: self.merges[item])

            token_list = self.merge_op(
                token_list,
                pair,
                self.merges[pair]
            )

        return token_list
    def encode(self, text:str):
        chunks = self.pre_tokenization(text)
        encoded = []

        for chunk in chunks:
            encoded.extend(self.encode_chunk(chunk))
        return encoded
            
    def decode(self, token_ids:list[int]):
        decoded_string = b""

        decoded_list = []
        for id in token_ids:
            if id in self.vocab:
                decoded_string += self.vocab[id]
            else:
                raise ValueError(f"Invalid ID: {id}")

        return decoded_string.decode()

    
    def save(self, filepath=None):
        if filepath is None:
            filepath = Path(__file__).resolve().parent / "merges.json"

        filepath = Path(filepath)

        data = {
            "merges": [
                [pair[0], pair[1], new_id]
                for pair, new_id in self.merges.items()
            ]
        }

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def load(self, filepath=None):
        if filepath is None:
            filepath = Path(__file__).resolve().parent / "merges.json"

        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Reset existing state
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Reconstruct merge rules and vocabulary
        for token1, token2, new_id in data["merges"]:
            pair = (token1, token2)

            self.merges[pair] = new_id
            self.vocab[new_id] = (
                self.vocab[token1] + self.vocab[token2]
            )

