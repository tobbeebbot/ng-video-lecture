from base import Tokenizer, get_stats, merge
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
class MyTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def _count(self, utf8_text, pair_counts = None, multiplier = 1):
        pair_counts = {} if pair_counts == None else pair_counts
        for pair in zip(utf8_text, utf8_text[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + multiplier
        return pair_counts

    def _merge(self, utf8_text, pair, new_token):
        utf8_new = []
        i = 0
        p0, p1 = pair
        while i < len(utf8_text):
            if i < len(utf8_text) - 1 and utf8_text[i] == p0 and utf8_text[i + 1] == p1:
                utf8_new.append(new_token)
                i += 2
            else:
                utf8_new.append(utf8_text[i])
                i += 1
        return utf8_new

    def train(self, text, vocab_size, verbose=False):

        
        def max_pair(pair_counts):
            pair, _val = max(pair_counts.items(), key= lambda x: x[1])
            return pair

        
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        num_merges = vocab_size - 256

        pattern = re.compile(GPT4_SPLIT_PATTERN)
        text_chunks = re.findall(pattern, text)
        print("Total chunks: ", len(text_chunks))
        text_chunks_counted = {}
        for tc in text_chunks:
            text_chunks_counted[tc] = text_chunks_counted.get(tc, 0) + 1

        utf_chunks_with_count = list(map(lambda item: (list(item[0].encode("utf-8")), item[1]), text_chunks_counted.items()))
        print("Uniquie chunks: ", len(utf_chunks_with_count))

        merges = {}
        #utf8_text = list(text.encode("utf-8"))
        for i in range(num_merges):
            next_token = 256 + i
            stats = {}
            for chunk, mult in utf_chunks_with_count:
                self._count(chunk, stats, mult)
            
            pair = max_pair(stats)
            merges[pair] = next_token
            vocab[next_token] = vocab[pair[0]] + vocab[pair[1]]

            # utf8_chunks = [merge(utf8_text, pair, next_token) for utf8_text in utf8_chunks]
            # utf8_chunks = list(filter(lambda chunk: len(chunk) > 1, utf8_chunks))
            utf_chunks_with_count = list(
                map(lambda chunk_count: (self._merge(chunk_count[0], pair, next_token), chunk_count[1]),
                    utf_chunks_with_count
                )
            )
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {next_token} ({vocab[next_token]}) had {stats[pair]} occurrences")
        
        self.vocab = vocab
        self.merges = merges
        

    def decode(self, ids):
        detokenized = b"".join(self.vocab[int(i)] for i in ids)
        return detokenized.decode("utf-8", errors="replace")

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = self._count(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(self, text):
        pattern = re.compile(GPT4_SPLIT_PATTERN)
        text_chunks = re.findall(pattern, text)
        encoded = []
        for chunk in text_chunks:
            chunk_bytes = list(chunk.encode("utf-8"))
            ids = self._encode_chunk(chunk_bytes)
            encoded.extend(ids)

        return encoded
        
