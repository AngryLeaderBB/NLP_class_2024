from time import time


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    @staticmethod
    def get_stats(tokens: list[int]) -> tuple[dict[tuple[int, int], int],
                                              tuple[int, int]]:
        # Initialize variables
        max_pair = (0, 0)
        max_count = 0
        pair_counter = {}

        # For each pair (tokens[i], tokens[i+1])
        for pair in zip(tokens, tokens[1:]):
            # Increase pair counter
            pair_counter[pair] = pair_counter.get(pair, 0) + 1
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

    def decode(self, tokens: list[int]) -> str:
        # Get byte-stream from tokens
        text_bytes = b"".join(self.vocab[t] for t in tokens)
        # Decode byte-stream into text (considering error of utf-8 representation)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        # Initialize tokens
        tokens = [i for i in text.encode("utf-8")]

        n = len(tokens)
        for _ in range(n):
            if len(tokens) < 2:
                break
            # Get counter dict (pair->num_repetition)
            counter, _ = Tokenizer.get_stats(tokens)
            # Minimum valid pair
            pair = min(counter, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            # Get byte id
            idx = self.merges[pair]
            # Update tokens
            tokens = Tokenizer.merge_pair(tokens, pair, idx)
        # Return tokens
        return tokens

    def train(self, text: str, vocab_size: int, debug=False) -> None:
        # Initialize variables
        tokens = [i for i in text.encode("utf-8")]
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        num_merges = vocab_size-256

        for i in range(num_merges):
            # Get time for debug
            if debug:
                last_time = time()

            # Get maximum pair (considering counter)
            _, pair = Tokenizer.get_stats(tokens)

            # New id
            idx = 256+i

            # Update vocabulary and merges
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # Update tokens
            tokens = Tokenizer.merge_pair(tokens, pair, idx)

            # -------- Debug ---------
            if debug:
                num_bars = 50 * (i + 1) // num_merges
                print(f"[{'-' * num_bars}{' ' * (50 - num_bars)}] {100 * (i + 1) / num_merges}%,"
                      f" time = {time()-last_time}")

        # Save vocabulary and merges
        self.vocab = vocab
        self.merges = merges

    def text_to_tokens(self, text: str) -> str:
        string_buffer = []
        # Encode text
        tokens = self.encode(text)
        # Create string separating tokens
        for t in tokens:
            string_buffer.append("[ ")
            string_buffer.append(self.vocab[t].decode("utf-8", errors="replace"))
            string_buffer.append(" ]")
        return "".join(string_buffer)

if __name__ == "__main__":
    string = "agdsvavbavavavd"
    a = Tokenizer()
    a.train(string, 258)

    print(a.vocab)
    print(a.merges)

    b = a.encode(string)
    print(b)
    print(a.decode(b))
