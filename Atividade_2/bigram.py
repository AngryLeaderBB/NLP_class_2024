from collections.abc import Callable
from typing import Any
from random import randint
import torch

class Bigram:
    @staticmethod
    def generate_token_dicts(token_list: list) -> tuple[dict, dict]:
        # Get the tokens (not considering duplicates)
        tokens = set(token_list)

        # Dict from tokens to index
        token_to_idx = {val:idx for idx, val in enumerate(tokens)}
        # Dict from index to tokens
        idx_to_token = {idx:val for val, idx in token_to_idx.items()}

        # Return both dicts
        return token_to_idx, idx_to_token

    @staticmethod
    def generate_probabilities(token_list: list, token_to_index: dict, fill_val: int = 0) -> torch.tensor:
        # Initialize counter tensor (int 64)
        vocab_size = len(token_to_index)
        counter_tensor = torch.full((vocab_size, vocab_size), fill_val, dtype=torch.int64)

        # Count token pairs
        for t1, t2 in zip(token_list, token_list[1: ]):
            counter_tensor[token_to_index[t1], token_to_index[t2]] += 1

        # Convert to float 64
        probs = counter_tensor.to(torch.float64)

        # Divide each row by the sum of the entry of itself
        probs /= torch.sum(probs, 1, keepdim=True)

        # Return probabilities
        return probs


    def __init__(self, encoder: Callable[[str], list[Any]],
                       decoder:Callable[[Any], str] = lambda x: x):

        self.encoder = encoder # Encoder function (string to token list)
        self.decoder = decoder  # Decoder function (token to string)
        self.token_to_idx = {}  # Token to index dict
        self.idx_to_token = {}  # Index to token dict
        self.probabilities = torch.zeros([]) # Probabilities tensor


    def train(self, base_text: str, fill_val: int = 0):
        # Generate token list
        token_list = self.encoder(base_text)

        # Generate token to index and index to token dicts
        self.token_to_idx, self.idx_to_token = Bigram.generate_token_dicts(token_list)

        # Set probability tensor
        self.probabilities = Bigram.generate_probabilities(token_list, self.token_to_idx, fill_val)

    def generate_text(self, max_tokens: int, seed: int=-1) -> str:
        # If you have a seed, you use on the torch pseudo number generator
        if seed == -1:
            g = torch.Generator().manual_seed(randint(0, 2**32 - 1))
        else:
            g = torch.Generator().manual_seed(seed)

        # Generate string buffer
        buffer = []
        # Get random initial index
        idx = torch.randint(0, len(self.token_to_idx), (1,), generator=g).item()

        for _ in range(max_tokens):
            # Add to buffer the decoding of the index
            buffer.append(self.decoder(self.idx_to_token[idx]))

            # if row equal to nan break it
            if torch.isnan(self.probabilities[idx][0]):
                break

            # Gets next index based on next probabilities
            idx = torch.multinomial(self.probabilities[idx],
                                    num_samples=1, replacement=True, generator=g).item()

        # Join the string
        return "".join(buffer)


    def perplexity(self, word: str) -> float:
        # Tokens of the word
        tokens = self.encoder(word)

        # List of probabilities: [P(w_1), P(w_2| w_1), P(w_3| w_2), ..., P(w_n| w_{n-1})]
        word_prob = torch.tensor([1/len(self.token_to_idx)] +
                                 [self.probabilities[self.token_to_idx[t1], self.token_to_idx[t2]]
                                    for t1, t2 in zip(tokens, tokens[1: ])])

        # Assert length word probabilities to total number of tokens
        assert len(word_prob) == len(tokens)

        # Sum of the log of probabilities: \sum_{i=1}^{n} P(w_i| w_{i-1})
        sum_log = torch.sum(torch.log(word_prob))

        # Perplexity of the word: e^{-sum_log/n} = (\prod_{i=1}^{n} P(w_i| w_{i-1}))^{-1/n}
        return torch.exp(-sum_log/len(word)).item()



if __name__ == "__main__":
    text = "bacsbvasddsbdasbasvfsdfavdabvadb"

    bigram = Bigram(lambda x: list(x))
    bigram.train(text)

    print(bigram.token_to_idx)
    print(bigram.idx_to_token)
    print(bigram.probabilities)
    print(bigram.perplexity("vasbas"))
    print(bigram.generate_text(10))
