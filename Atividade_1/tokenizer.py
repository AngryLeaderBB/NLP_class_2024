from collections import defaultdict


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    @staticmethod
    def get_stats(tokens: list[int]) -> (defaultdict[int], (int, int)):
        # Initialize variables
        max_pair = (0, 0)
        max_count = 0
        pair_counter = defaultdict(int)

        # For each pair (tokens[i], tokens[i+1]
        for pair in zip(tokens, tokens[1:]):
            # Increase pair counter
            pair_counter[pair] += 1
            count = pair_counter[pair]

            # Update max pair
            if max_count < count:
                max_count = count
                max_pair = pair

        # Return max pair
        return pair_counter, max_pair

    @staticmethod
    def merge_pair(tokens: list[int], new_pair: tuple[int], id: int) -> list[int]:
        # Initialize variables
        res = []
        skip = False

        # For each pair
        for pair in zip(tokens, tokens[1:]):
            # If new pair found, skip one append command
            if skip:
                skip = False
                continue

            # If new pair found append id
            if new_pair == pair:
                res.append(id)
                skip = True
            # Else append tokens[i]
            else:
                res.append(pair[0])

        # If last element must be skipped, skip it
        if skip:
            return res

        # Else append the last element
        res.append(tokens[-1])
        # Returns a new tokens
        return res

    def train(self, text: str, vocab_size: int) -> None:
        # Initialize variables
        num_merges = vocab_size-256
        tokens = [i for i in text.encode("utf-8")]
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            # Get maximum pair (considering counter)
            _, pair = Tokenizer.get_stats(tokens)

            # New id
            id = 256+i

            # Update vocabulary and merges
            merges[pair] = id
            vocab[id] = vocab[pair[0]] + vocab[pair[1]]

            # Update tokens
            tokens = Tokenizer.merge_pair(tokens, pair, id)

        # Save vocabulary and merges
        self.vocab = vocab
        self.merges = merges

    def decode(self, tokens: list[int]) -> str:
        # Get byte-stream from tokens
        text_bytes = b"".join(self.vocab[t] for t in tokens)
        # Decode byte-stream into text (considering error of utf-8 representation)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        # Initialize tokens
        tokens = [i for i in text.encode("utf-8")]

        while len(tokens) >= 2:
            # Get counter dict (pair->num_repetition)
            counter, _ = Tokenizer.get_stats(tokens)
            # Minimum valid pair
            pair = min(counter, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            # Get byte id
            id = self.merges[pair]
            # Update tokens
            tokens = Tokenizer.merge_pair(tokens, pair, id)
        # Return tokens
        return tokens

if __name__ == "__main__":
    string = "agdsvavbavavavd"
    a = Tokenizer()
    a.train(string, 258)

    print(a.vocab)
    print(a.merges)

    b = a.encode(string)
    print(b)
    print(a.decode(b))

