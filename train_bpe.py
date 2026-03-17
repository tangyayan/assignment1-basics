import regex
import os

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import functools
from functools import reduce
import heapq
from collections import defaultdict, Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries

import pickle #用于保存预分词结果
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
token_re = regex.compile(PAT)
# def _pretokenize_chunk(args) -> Counter:
#     input_path, start, end, special_tokens = args
#     # import psutil
#     # process = psutil.Process()
#     # chunk_id = f"{start}-{end}"
#     # print(f"[{chunk_id}] 开始,内存: {process.memory_info().rss / 1024 / 1024:.2f} MB")

#     with open(input_path, "rb") as f:
#         f.seek(start)
#         chunk_bytes = f.read(end - start)
    
#     text = chunk_bytes.decode("utf-8", errors="replace")
#     del chunk_bytes
#     counter = Counter()

#     if not special_tokens:
#         tokens = [m.group(0) for m in token_re.finditer(text)]#批量提取
#         encoded_tokens = [tuple(bytes([b]) for b in t.encode("utf-8")) for t in tokens]
#         counter.update(encoded_tokens)
#         return counter
    
#     # print(f"[{chunk_id}] 处理中,内存: {process.memory_info().rss / 1024 / 1024:.2f} MB")
#     # print("begin pretokenize chunk")

#     parts = text.split("<|endoftext|>")
#     print("end split, begin pretokenize special tokens")
#     del text
#     all_tokens = []
#     # print(len(parts),len(parts[0]))
#     i = 0
#     for part in parts:
#         if not part:
#             continue
#         # all_tokens.extend(m.group(0) for m in token_re.finditer(part))
#         # all_tokens.extend(token_re.findall(part))

#         now_tokens = []
#         for t in token_re.findall(part):
#             now_tokens.append(tuple(bytes([b]) for b in t.encode("utf-8")))
#         counter.update(now_tokens)
#         if i%1000 == 0:
#             print(f"pretokenized {i} parts")
#         i += 1

#     # print(f"[{chunk_id}] 结束,内存: {process.memory_info().rss / 1024 / 1024:.2f} MB")
#     print("end pretokenize chunk")

#     # encoded_tokens = [tuple(bytes([b]) for b in t.encode("utf-8")) for t in all_tokens]
#     # counter.update(encoded_tokens)

#     return counter

def _pretokenize_chunk(chuck: tuple[int],
                  input_path: str,
                  special_tokens: list[str]) -> Counter:
    """
    Process each chunk of the file and update the vocabulary counter.
    """
    start, end = chuck

    special_tokens_pattern = '|'.join(special_tokens)
    # special_tokens_pattern = "|".join(map(re.escape, special_tokens))
    chunk_counter = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="replace")
        # 2.1 在预分词前移除特殊标记
        for segment in regex.split(special_tokens_pattern, chunk):
            # 3. 预分词(pre-tokenization)
            for match in regex.finditer(PAT, segment):
                token = match.group()
                if token:
                    chunk_counter.update([tuple(bytes([b]) for b in token.encode("utf-8"))])
    return chunk_counter

class heap_elm:
    def __init__(self, count, pair):
        self.count = count
        self.pair = pair
    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        counters_file: 用于预加载counter文件，如果存在则跳过预分词步骤
        pretoken_file: 用于保存预分词结果的文件路径，如果提供了该参数，则会将预分词得到的 counter 保存到该文件中，以便后续加载使用

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    num_processes = kwargs.get("num_processes", 8)
    num_split = 1024
    counters_file = kwargs.get("counters_file", "counters.pkl") 
    pretoken_file = kwargs.get("pretoken_file", None)

    start = time.time()

    if os.path.exists(counters_file):
        print(f"Loading counters from {counters_file}...")
        with open(counters_file, "rb") as f:
            global_counter = pickle.load(f)
    else:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_split, b"<|endoftext|>")

            if len(boundaries) <= 2:
                boundaries = find_chunk_boundaries(f, num_split, b" ")
                print("Using newline as chunk boundary.")
        
        """线程级"""
        # args_list = [
        #     (input_path, boundaries[i], boundaries[i+1], special_tokens)
        #     for i in range(len(boundaries) - 1)
        # ]
        # print("Starting pre-tokenization...")
        # with ThreadPoolExecutor(max_workers=num_processes) as executor:
        #     counters = list(executor.map(_pretokenize_chunk, args_list))

        # # global_counter = Counter()
        # # for c in counters:
        #     # global_counter.update(c)
        # print("end pre-tokenization. Starting BPE merging...")
        # global_counter = reduce(operator.add, counters, Counter())

        """进程级"""
        global_counter = Counter()
        print("Starting pre-tokenization...")
        with ThreadPool(num_processes) as pool:
            results = pool.imap_unordered(
                    functools.partial(_pretokenize_chunk, input_path=input_path, special_tokens=special_tokens),
                    zip(boundaries[:-1], boundaries[1:]),
                )

            for res in results:
                global_counter.update(res)
        
        print("end pre-tokenization. Starting BPE merging...")
        
        if pretoken_file:
            with open(pretoken_file, "wb") as f:
                pickle.dump(global_counter, f)
    token_list = list(global_counter.items())
    # # [((b't',b'h',b'e'), 2), ...]
    end = time.time()
    print(f"end merging, used time: {end - start:.2f}s")

    start = time.time()

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    now_voab_size = len(vocab)

    pair_counter = defaultdict(lambda: [0, set()])# {pair: [count, set(token_idx)]}
    heap = []
    for i, (token, count) in enumerate(token_list):
        for j in range(len(token) - 1):
            pair = (token[j], token[j + 1])
            pair_counter[pair][0] += count
            pair_counter[pair][1].add(i)
    for pair, (count, _) in pair_counter.items():
        h = heap_elm(count, pair)
        heapq.heappush(heap, h)#大根堆
    merges = []

    for step in range(vocab_size - now_voab_size):
        if not pair_counter:
            break
        # best_pair, (_, occ_set) = max(pair_counter.items(), key=lambda x: (x[1][0],x[0]))
        # """
        while heap:
            h = heapq.heappop(heap)
            count, best_pair = h.count, h.pair
            if best_pair in pair_counter and count == pair_counter[best_pair][0]:
                occ_set = pair_counter[best_pair][1]
                break
        # """
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        merges.append(best_pair)

        new_pairs = {}
        for i in occ_set.copy():
            token, count = token_list[i]
            new_token_list = []
            j = 0
            old_pair_list = []
            while j < len(token):
                if j < len(token) - 1 and (token[j], token[j+1]) == best_pair:
                    if(j > 0):
                        old_pair = (token[j-1], token[j])
                        pair_counter[old_pair][0] -= count
                        if(pair_counter[old_pair][0] == 0):
                            del pair_counter[old_pair]
                        else:
                            new_pairs[old_pair] = pair_counter[old_pair][0]
                            old_pair_list.append(old_pair)
                        new_pair = (token[j-1], new_token)
                        pair_counter[new_pair][0] += count
                        pair_counter[new_pair][1].add(i)
                        new_pairs[new_pair] = pair_counter[new_pair][0]

                    if(j < len(token) - 2):
                        old_pair = (token[j+1], token[j+2])
                        pair_counter[old_pair][0] -= count
                        if(pair_counter[old_pair][0] == 0):
                            del pair_counter[old_pair]
                        else:
                            new_pairs[old_pair] = pair_counter[old_pair][0]
                            old_pair_list.append(old_pair)
                        new_pair = (new_token, token[j+2])
                        pair_counter[new_pair][0] += count
                        pair_counter[new_pair][1].add(i)
                        new_pairs[new_pair] = pair_counter[new_pair][0]

                    new_token_list.append(new_token)
                    j += 2
                else:
                    new_token_list.append(token[j])
                    j += 1
        
            token_list[i] = (tuple(new_token_list), count)
            new_pairs_list = list(zip(new_token_list[:-1], new_token_list[1:]))
            for pair in old_pair_list:
                if pair not in new_pairs_list:
                    pair_counter[pair][1].discard(i)

        for pair, count in new_pairs.items():
            h = heap_elm(count, pair)
            heapq.heappush(heap, h)
        del pair_counter[best_pair]
    
    end = time.time()
    print(f"end BPE merging, used time: {end - start:.2f}s")
    
    # print(f"Pre-tokenization time: {mid1_time - start_time:.2f}s")
    # print(f"BPE training time: {end_time - mid1_time:.2f}s")
    return vocab, merges